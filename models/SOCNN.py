import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import weights_init

class SOCNN(nn.Module):
    def __init__(self, FLAGS):
        super(SOCNN, self).__init__()

        self.num_layers = {
            'sig':FLAGS.num_sig,
            'off':FLAGS.num_off
        }
        self.conv_filters = {
            'sig':FLAGS.conv_sig,
            'off':FLAGS.conv_off
        }

        self.len_input = FLAGS.len_input
        self.len_output = FLAGS.len_output
        self.dim_input = FLAGS.dim_input
        self.dim_output = FLAGS.dim_output

        for key in self.conv_filters.keys():
            if len(self.conv_filters[key]) == 1:
                self.conv_filters[key] *= self.num_layers[key]
            else:
                assert len(self.conv_filters[key]) == self.num_layers[key]

        # defining significance network
        sig_net = [] 
        n_input = self.dim_input
        for i in range(self.num_layers['sig']):
            n_output = self.conv_filters['sig'][i]
            if i == self.num_layers['sig'] - 1:
                n_output = self.dim_output

            ks = 3 if i % 2 == 0 else 1
            pad = 1 if ks == 3 else 0

            sig_net.append( nn.Sequential( nn.Conv1d(n_input, n_output, ks, padding=pad),
                                           nn.BatchNorm1d(n_output),
                                           nn.LeakyReLU(0.1, inplace=True) )
            )
            n_input = n_output
        self.sig_net = nn.Sequential( *sig_net )
                
        # defining offset network
        off_net = []
        n_input = self.dim_input
        for i in range(self.num_layers['off']):
            n_output = self.conv_filters['off'][i]
            if i == self.num_layers['off'] - 1:
                n_output = self.dim_output

            off_net.append( nn.Sequential( nn.Conv1d(n_input, n_output, 1), 
                                           nn.BatchNorm1d(n_output),
                                           nn.LeakyReLU(0.1, inplace=True) ) 
            )
            n_input = n_output 
        self.off_net = nn.Sequential( *off_net )
        
        self.reg_W = torch.randn((self.dim_output, self.len_input), requires_grad=True).cuda()


    def forward(self, x, aux):
        sig_x = self.sig_net(x)
        sig_x = F.softmax(sig_x, dim=0)
        off_x = self.off_net(x)

        # aux_output = torch.Tensor(aux)
        aux_output = torch.add(aux, off_x)
        # should rewrite auxiliary output part : different from paper !
        # aux_output = torch.sum(aux_output, 2).view(-1, self.dim_output)

        x = torch.mul(sig_x, off_x)
        #x = torch.sum(torch.mul(self.reg_W.expand_as(x).cuda(), x), 2)
        x = torch.sum(torch.mul(self.reg_W, x), 2)

        return x, aux_output


if __name__ == '__main__':
    pass
