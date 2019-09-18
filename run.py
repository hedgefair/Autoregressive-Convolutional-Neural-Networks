import os 
import yaml

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import DataSet
from utils import argparser, logging
from models.SOCNN import SOCNN


def train(model, FLAGS):
    model.train()

    mseloss = nn.MSELoss()

    step = 0 
    total_loss = 0
    total_loss_out = 0
    total_loss_aux = 0

    for idx, (data, target, aux) in enumerate(FLAGS.train_data.iter_once(FLAGS.batch_size)):
        data, target, aux = torch.tensor(data, device=FLAGS.device), \
                            torch.tensor(target, device=FLAGS.device), \
                            torch.tensor(aux, device=FLAGS.device)
        output, aux_output = model(data, aux)
        loss_out = mseloss(output, target)     

        aux_target = target.view(-1, FLAGS.dim_output, 1)
        loss_aux = ((aux_output - aux_target)**2).mean()
        
        total_loss_out += loss_out.item()
        total_loss_aux += loss_aux

        loss = loss_out + loss_aux * FLAGS.aux_weight
        total_loss += loss.item()

        FLAGS.optimizer.zero_grad()
        loss.backward()

        FLAGS.optimizer.step()
        step += 1

    total_loss /= step
    total_loss_out /= step
    total_loss_aux /= step
    
    FLAGS.error_train_out.append(total_loss_out)
    FLAGS.error_train_aux.append(total_loss_aux)

    logging("Epoch : {:3d} - loss_val : {:.4f}".format(FLAGS.epoch, total_loss), FLAGS.log_train)
    logging("Epoch : {:3d} - loss_out : {:.4f}".format(FLAGS.epoch, total_loss_out), FLAGS.log_train)
    logging("Epoch : {:3d} - loss_aux : {:.4f}".format(FLAGS.epoch, total_loss_aux), FLAGS.log_train)
    print("Train Epoch : {:3d} - loss_val : {:.4f}".format(FLAGS.epoch, total_loss))


def test(model, FLAGS):
    model.eval()

    mseloss = nn.MSELoss()

    step = 0 
    total_loss = 0
    total_loss_out = 0
    total_loss_aux = 0
    
    with torch.no_grad():
        for idx, (data, target, aux) in enumerate(FLAGS.test_data.iter_once(FLAGS.batch_size)):
            data, target, aux = torch.tensor(data, device=FLAGS.device), \
                                torch.tensor(target, device=FLAGS.device), \
                                torch.tensor(aux, device=FLAGS.device)
            output, aux_output = model(data, aux)
            loss_out = mseloss(output, target)     

            aux_target = target.view(-1, FLAGS.dim_output, 1)
            loss_aux = ((aux_output - aux_target)**2).mean()
            
            total_loss_out += loss_out.item()
            total_loss_aux += loss_aux

            loss = loss_out + loss_aux * FLAGS.aux_weight
            total_loss += loss.item()
            step += 1

    total_loss /= step
    total_loss_out /= step
    total_loss_aux /= step

    FLAGS.error_test_out.append(total_loss_out)
    FLAGS.error_test_aux.append(total_loss_out)

    logging("Epoch : {:3d} - loss_val : {:.4f}".format(FLAGS.epoch, total_loss), FLAGS.log_test)
    logging("Epoch : {:3d} - loss_out : {:.4f}".format(FLAGS.epoch, total_loss_out), FLAGS.log_test)
    logging("Epoch : {:3d} - loss_aux : {:.4f}".format(FLAGS.epoch, total_loss_aux), FLAGS.log_test)
    print("Test Epoch : {:3d} - loss_val : {:.4f}".format(FLAGS.epoch, total_loss))


def main():
    FLAGS, _ = argparser()

    with open(FLAGS.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    FLAGS.epoch = 0
    FLAGS.batch_size = config['run']['batch_size']
    FLAGS.aux_weight = config['run']['aux_weight']

    FLAGS.error_train_out = []
    FLAGS.error_train_aux = []
    FLAGS.error_test_out = []
    FLAGS.error_test_aux = []

    # build dataset files
    FLAGS.train_data = DataSet(config['run']['train_data'], need_shuffle=True)
    FLAGS.test_data = DataSet(config['run']['test_data'], need_shuffle=False)
    FLAGS.val_data = DataSet(config['run']['val_data'], need_shuffle=True)

    # set path to log file
    FLAGS.log_train = config['run']['log_train']
    FLAGS.log_test = config['run']['log_test'] 
    FLAGS.log_error = config['run']['log_error']

    # set network hyperparameter
    FLAGS.len_input = config['network']['len_input']
    FLAGS.dim_input = config['network']['dim_input']
    FLAGS.len_output = config['network']['len_output']
    FLAGS.dim_output = config['network']['dim_output']
    FLAGS.num_sig = config['network']['num_sig']
    FLAGS.num_off = config['network']['num_off']
    FLAGS.conv_sig = config['network']['conv_sig']
    FLAGS.conv_off = config['network']['conv_off']
    
    # construct model 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['run']['cuda_devices'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    FLAGS.device = device

    MODEL = SOCNN
    model = MODEL(FLAGS).to(device)
    if FLAGS.load_model is not None:
        model.load_state_dict(FLAGS.load_model)

    FLAGS.optimizer = optim.Adam(model.parameters(), lr=config['run']['learning_rate'])

    for i in range(config['run']['epoch']):
        FLAGS.epoch += 1
        
        train(model, FLAGS)
        test(model, FLAGS)

        if (i+1) % 10 == 0:
            torch.save(model.state_dict(), config['run']['model_path'])

    df = pd.DataFrame( {'train_out':FLAGS.error_train_out,
                        'train_aux':FLAGS.error_train_aux,
                        'test_out':FLAGS.error_test_out,
                        'test_aux':FLAGS.error_test_aux} )
    df.to_csv(FLAGS.log_error)

if __name__ == '__main__':
    main()

