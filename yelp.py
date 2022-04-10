import torch
from torch import nn
from collections import OrderedDict

class YelpModelv0(nn.Module):
    def __init__(self,
                 embedding,
                 int_dim = 4,
                 int_layers_param=(128,256),

                 text_dim = 2,
                 lstm_hidden_dim=200,
                 num_lstm_layers=2,
                 text_layers_param=(128, 32),

                 concat_layers_param = (128,32),
                 dropout=0.1,
                 fix_embedding=True):
        super(YelpModelv0, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)
        # Whether fix embedding
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)

        self.text_lstm = nn.LSTM(self.embedding_dim, lstm_hidden_dim, num_layers=num_lstm_layers, dropout=dropout,batch_first=True)
        text_layers = [nn.Linear(lstm_hidden_dim,text_layers_param[0]), nn.BatchNorm1d(text_layers_param[0])]
        for i in range(len(text_layers_param)-1):
            text_layers.append(nn.Linear(text_layers_param[i],text_layers_param[i+1]))
            text_layers.append(nn.BatchNorm1d(text_layers_param[i+1]))
        self.text_layers = nn.Sequential(*text_layers)

        int_layers = [nn.Linear(int_dim,int_layers_param[0]),nn.BatchNorm1d(int_layers_param[0])]
        for i in range(len(int_layers_param)-1):
            int_layers.append(nn.Linear(int_layers_param[i],int_layers_param[i+1]))
            int_layers.append(nn.BatchNorm1d(int_layers_param[i+1]))
        self.int_layers = nn.Sequential(*int_layers)

        concat_layers = [nn.Linear(int_layers_param[-1]+text_dim*text_layers_param[-1],concat_layers_param[0])]
        for i in range(len(concat_layers_param)-1):
            concat_layers.append(nn.Linear(concat_layers_param[i],concat_layers_param[i+1]))
        concat_layers.append(nn.Linear(concat_layers_param[-1], 1))
        concat_layers.append(nn.Sigmoid())
        self.concat_layers = nn.Sequential(*concat_layers)

    def forward(self,int_array, class_array, text_embedding_array):
        text_embedding_array = self.embedding(text_embedding_array)
        x_int = self.int_layers(int_array)

        x_texts = []
        for i in range(text_embedding_array.shape[1]):
            text_embedding_array_sub = text_embedding_array[:,i]
            x_text_tmp,_ = self.text_lstm(text_embedding_array_sub,None)
            x_text_tmp = x_text_tmp[:,-1]
            x_text_tmp = self.text_layers(x_text_tmp)
            x_texts.append(x_text_tmp)
        x_texts = torch.concat(x_texts,dim=-1)

        x_concat = torch.concat((x_int,x_texts),dim=-1)

        x = self.concat_layers(x_concat)

        return x
