import pickle

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def read_uci(FLAGS):
    fpath = FLAGS.fpath

    data_df = pd.read_csv( fpath, header=0, sep=';', index_col='date_time',
                           parse_dates={'date_time':[0,1], 'date':[0], 'time':[1]} )
    print("Data Read-in completed")
    columns = data_df.columns

    # get list of date and time in the time stamps
    list_date = [str(date).split()[0] for date in data_df['date']]
    list_time = [str(time).split()[1] for time in data_df['time']]
    
    data_df.drop(['date', 'time'], axis=1, inplace=True)

    # remove non-numeric/nan values in the dataframe
    for column in columns:
        if column == 'date' or column =='time':
            continue
        data_df[column] = pd.to_numeric(data_df[column], errors='coerce')
    data_df = data_df.loc[data_df.notnull().all(axis=1)] 

    encDate = LabelEncoder()
    encTime = LabelEncoder()
    encDate.fit(list_date)
    encTime.fit(list_time)
    
    # set train and test data portion
    idx_train = int(len(data_df) * 0.6)
    idx_val = int(len(data_df) * 0.8)

    FLAGS.data_train = data_df.iloc[:idx_train]
    FLAGS.data_val = data_df.iloc[idx_train:idx_val]
    FLAGS.data_test = data_df.iloc[idx_val:]

    FLAGS.data_df = data_df
    FLAGS.encDate = encDate
    FLAGS.encTime = encTime

    FLAGS.feature_cols = ['feature_{}'.format(i) for i in range(7)]
    FLAGS.cols = ['Date', 'Time', 'value'] + FLAGS.feature_cols

    FLAGS.target_cols = [0, 1, 2, 3, 4, 5, 6]


def build_uci(FLAGS):
    print("Generating data points for the model")
    data_dict = {}
    data_dict['data'], data_dict['val'], data_dict['aux'] = generate_data(FLAGS.data_train, 1000, 100)
    with open('data_files/uci_train.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle)
    del data_dict 

    data_dict = {}
    data_dict['data'], data_dict['val'], data_dict['aux'] = generate_data(FLAGS.data_val, 100, 100)
    with open('data_files/uci_test.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle)
    del data_dict 

    data_dict = {}
    data_dict['data'], data_dict['val'], data_dict['aux'] = generate_data(FLAGS.data_test, 100, 100)
    with open('data_files/uci_val.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle)
    del data_dict 
    

def generate_data(data, n_sample, data_per_sample):
    length = FLAGS.len_output + FLAGS.len_input
    data_list = []
    target_list = []
    aux_list = []
    
    for i in range(n_sample):
        async_val, sampled = sample_uci(data, FLAGS)
        async_val = async_val.astype(float)

        for _ in range(data_per_sample):
            idx = np.random.randint(0, len(async_val) - FLAGS.len_input - FLAGS.len_output)            
            data_val = async_val.iloc[idx:idx + FLAGS.len_input][FLAGS.cols]
            data_list.append(data_val.values.transpose(1, 0))
            target_list.append(sampled.iloc[idx + FLAGS.len_input, FLAGS.target_cols].values)

            vals = async_val.iloc[idx:idx + FLAGS.len_input,2].values.reshape(-1, 1)
            temp = async_val.iloc[idx:idx + FLAGS.len_input,3:].values
            aux_list.append(np.multiply(vals, temp).transpose())
        
        print("Reading in have done for {} files".format(i))
        
    return np.array(data_list, dtype=np.float32), np.array(target_list, dtype=np.float32), \
            np.array(aux_list, dtype=np.float32)


def sample_uci(data_df, FLAGS):
    encDate = FLAGS.encDate
    encTime = FLAGS.encTime
    
    # sampling time series
    indices = sampling_points(len(data_df), 1, 8)
    sampled = FLAGS.data_df.iloc[indices]

    # sampling one attributes at a time
    async_val = []
    cols = ['Date', 'Time', 'feature', 'value']
    
    for idx, date in enumerate(sampled.index):
        observed = [str(date.date()), str(date.time())]

        x = np.random.randint(0, sampled.shape[1])
        observed.extend([x, sampled.iloc[idx, x]])
        async_val.append(observed)
    
    async_val = pd.DataFrame(np.array(async_val), columns=cols)

    # encoding values in the dataframe : feature columns
    async_val = pd.get_dummies(async_val, columns=['feature'])
    async_val['Date'] = encDate.transform(list(async_val['Date']))
    async_val['Time'] = encTime.transform(list(async_val['Time']))

    async_val['Date'] = async_val['Date'].diff().fillna(0)
    async_val['Time'] = async_val['Time'].diff().fillna(0)

    #async_val['Date'] = async_val['Date']/async_val['Date'].max() * 3
    #async_val['Time'] = async_val['Time']/async_val['Time'].max() * 3

    return async_val, sampled


def sampling_points(length, min_int, max_int):
    last_index = -1
    indices = []

    while True :
        x = np.random.randint(1,8)
        last_index += x
        if last_index >= length:
            break
        else:
            indices.append(last_index)

    return indices


def argparser():
    parser = argparse.ArgumentParser()
    
    # path to the raw data file
    parser.add_argument(
        '--fpath',
        type=str,
        default=''
    )
    FLAGS, _ = parser.parse_known_args()

    return FLAGS, _


if __name__ == '__main__':
    FLAGS, _ = argparser()
    read_uci(FLAGS)
    build_uci(FLAGS)
