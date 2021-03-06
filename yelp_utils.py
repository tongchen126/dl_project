import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from gensim.models import word2vec
import re
import torch
from matplotlib import pyplot as plt
import datetime

from yelp import *
def pd_load_csv(file,cols=None):
    with open(file,'r') as f:
        return pd.read_csv(f,usecols=cols)

def pd_load_json(file,cols=None):
    with open(file,'r') as f:
        return pd.read_json(f, orient="records", lines=True)

def construct_dataset(business:pd.DataFrame,review:pd.DataFrame,covid:pd.DataFrame):
    business_covid = pd.merge(business,covid,on='business_id')
    return pd.merge(business_covid,review,on='business_id')

def train_word2vec(x,vector_size=250,save_path='model/yelp/w2v_yelp.model'):
    model = word2vec.Word2Vec(x, vector_size=vector_size, window=5, min_count=5, workers=12)
    if save_path:
        model.save(save_path)
    return model

def save_loss(data,file='model/yelp/loss.png'):
    iterations = range(0, len(data))
    plt.plot(iterations, data)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.ylim()
    plt.savefig(file)
    plt.close()

class Preprocess():
    def __init__(self, sen_len, w2v_path="model/yelp/w2v_yelp.model"):
        self.w2v_path = w2v_path
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    def get_w2v_model(self):
        # load word to vector model
        self.embedding = word2vec.Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    def add_embedding(self, word):
        # add word into embedding
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    def make_embedding(self, load=True):
        print("Get embedding ...")
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError

        for i, word in enumerate(self.embedding.wv.key_to_index):
            print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self,sentence):
        sentence_idx = []
        for word in sentence:
            if (word in self.word2idx.keys()):
                sentence_idx.append(self.word2idx[word])
            else:
                sentence_idx.append(self.word2idx["<UNK>"])
        sentence_idx = self.pad_sequence(sentence_idx)
        return torch.LongTensor(sentence_idx)

class YelpDataset(Dataset):
    USECOLS = [
        'city', 'state', 'latitude', 'longitude', 'stars_x', 'review_count', 'categories',
        'stars_y', 'date', 'text', 'useful',
        'highlights', 'delivery or takeout', 'Grubhub enabled', 'Call To Action enabled', 'Request a Quote Enabled', 'Covid Banner', 'Virtual Services Offered'
    ]
    CLASS_COLS = []
    INT_COLS = ['review_count','stars_x','stars_y','useful','latitude', 'longitude']
    INT_NOR = [100.0, 5.0, 5.0, 1,1000.0, 1000.0]
    INT_SHIFT = [0, 0.01, 0.01, 0, 0, 0]

    SPECIAL_INT_COLS = ['date']
    SPECIAL_INT_METHOD = [
        lambda x : datetime.date.fromisoformat(x).toordinal()/10e6,
    ]
    INT_COLS_DIM = len(INT_COLS) + len(SPECIAL_INT_COLS)
    TEXT_COLS = ['categories','text']
    COVID_COLS = ['highlights', 'delivery or takeout', 'Grubhub enabled', 'Call To Action enabled',  'Covid Banner']
    COVID_WEIGHTS = [3,1,2,2,3]
    def __init__(self,data_path,w2v:Preprocess=None,maxlen=1e6,make_embedding=True):
        self.raw_data = pd_load_csv(data_path,cols=self.USECOLS).iloc[:int(maxlen)]
        self.w2v = w2v
        self.data = []
        self.word2vec_dataset = []

        if w2v and make_embedding:
            self.embedding = w2v.make_embedding()
        else:
            self.embedding = None

        self.init_dataset()

    def init_dataset(self):
        print('Dataset length {0}'.format(len(self)))
        for i in range(len(self)):
            int_array = []
            class_array = []
            text_embedding_array = []
            covid_array = []

            row = self.raw_data.iloc[i]

            for i,col in enumerate(self.INT_COLS):
                int_array.append(float((row[col]+self.INT_SHIFT[i])/self.INT_NOR[i]))

            for i,col in enumerate(self.SPECIAL_INT_COLS):
                special=self.SPECIAL_INT_METHOD[i](row[col])
                int_array.append(special)

            for col in self.CLASS_COLS:
                class_array.append(int(row[col]))

            for col in self.TEXT_COLS:
                sentence = self.sentence_split(row[col])
                sentence_embedding = self.w2v.sentence_word2idx(sentence)
                text_embedding_array.append(sentence_embedding)

            for i,col in enumerate(self.COVID_COLS):
                col_value = row[col]
                col_value = str(col_value).upper() == 'FALSE'
                covid_result = 0 if col_value else self.COVID_WEIGHTS[i]
                covid_array.append(covid_result)
            covid_target = np.sum(covid_array)/np.sum(self.COVID_WEIGHTS)

            int_array = torch.FloatTensor(int_array)
            class_array = torch.IntTensor(class_array)
            text_embedding_array = torch.stack(text_embedding_array, dim=0)

            self.data.append((int_array, class_array, text_embedding_array, covid_target))
            if len(self.data) % 1000 == 0:
                print('Current length {0}'.format(len(self.data)))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        int_array, class_array, text_embedding_array, covid_target = self.data[idx]
        return (int_array,class_array,text_embedding_array,covid_target)

    def sentence_split(self,sentence):
        return list(filter(None, re.split(' |;|,|\.|&', sentence.strip('\n'))))

    def prepare_word2vec_data(self,maxlen=5e6):
        if len(self.word2vec_dataset):
            return self.word2vec_dataset
        for i in range(len(self)):
            texts = []
            row = self.raw_data.iloc[i]
            for col in self.TEXT_COLS:
                texts.append(row[col])
            self.word2vec_dataset.extend([self.sentence_split(line) for line in texts])
            print('prepare_word2vec_data: w2v dataset length {0}'.format(len(self.word2vec_dataset)))
            if len(self.word2vec_dataset) > maxlen:
                break
        return self.word2vec_dataset

if __name__ == '__main__':
    METHODS = ['train_w2v', 'inspect_dataset','inspect_model','train','eval']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    method = METHODS[4]
    if method == 'train_w2v':
        dataset = YelpDataset('/root/Downloads/yelp/kaggle/business_review_covid.csv')
        w2v_dataset = dataset.prepare_word2vec_data()
        train_word2vec(w2v_dataset)
    elif method == 'inspect_dataset':
        w2v_model = Preprocess(30)
        dataset = YelpDataset('/root/Downloads/yelp/kaggle/business_review_covid.csv',w2v_model)
        a=dataset[0]
        embedding = dataset.embedding
        emb_size = embedding.size()
    elif method == 'inspect_model':
        w2v_model = Preprocess(30)
        dataset = YelpDataset('/root/Downloads/yelp/kaggle/business_review_covid.csv',w2v_model,maxlen=1e6)
        model = YelpModelv0(dataset.embedding,len(dataset.INT_COLS)).to(device)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=32,shuffle=False,num_workers=0)
        for step, data in enumerate(train_loader):
            int_array, class_array, text_embedding_array = data[:3]
            y=data[3]

            int_array=int_array.to(device)
            class_array=class_array.to(device)
            text_embedding_array=text_embedding_array.to(device)
            y=y.to(device)

            model(int_array, class_array, text_embedding_array)
    elif method == 'train':
        config = {
            'w2v_length'    :   30,
            'dataset_length':   5e6,
            'batch_size' : 64,
            'num_workers': 8,
            'lr'         : 1e-3,
            'epoch'      : 10,
            'eval_interval':100,
        }
        w2v_model = Preprocess(config['w2v_length'])
        dataset = YelpDataset('/root/Downloads/yelp/kaggle/business_review_covid_part.csv',w2v_model,maxlen=config['dataset_length'])
        val_dataset = YelpDataset('/root/Downloads/yelp/kaggle/business_review_covid_val.csv',w2v_model,make_embedding=False)
        model = YelpModelv1(dataset.embedding,dataset.INT_COLS_DIM,text_dim=len(dataset.TEXT_COLS)).to(device)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=config['batch_size'],shuffle=True,num_workers=config['num_workers'])
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=0)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
        criterion = torch.nn.L1Loss()
        loss_list = []
        val_loss_list = []
        min_val = 1.0
        min_val_epoch = 0
        for epoch in range(config['epoch']):
            for step, data in enumerate(train_loader):
                optimizer.zero_grad()
                int_array, class_array, text_embedding_array = data[:3]
                y = data[3]

                int_array = int_array.to(device)
                class_array = class_array.to(device)
                text_embedding_array = text_embedding_array.to(device)
                y = y.to(device)

                pred = model(int_array, class_array, text_embedding_array).squeeze()
                loss = criterion(pred,y)
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                if step % config['eval_interval'] == 0:
                    model.eval()
                    val_loss_list_tmp = []
                    for step, data in enumerate(val_loader):
                        int_array, class_array, text_embedding_array = data[:3]
                        y = data[3]

                        int_array = int_array.to(device)
                        class_array = class_array.to(device)
                        text_embedding_array = text_embedding_array.to(device)
                        y = y.to(device)

                        pred = model(int_array, class_array, text_embedding_array).squeeze()
                        loss = criterion(pred, y)

                        val_loss_list_tmp.append(loss.item())

                    val_loss_list.append(np.mean(val_loss_list_tmp))
                    model.train()

                    print('epoch {0}, loss {1},val loss {2}'.format(epoch,np.mean(loss_list[config['eval_interval']:-1]),np.mean(val_loss_list_tmp)))
                    save_loss(loss_list,file='model/yelp/loss.png')
                    save_loss(val_loss_list,file='model/yelp/loss_val.png')
                    np.save('model/yelp/loss.npy',np.array(loss_list))
                    np.save('model/yelp/loss_val.npy',np.array(val_loss_list))
                    if np.mean(val_loss_list_tmp) < min_val:
                        min_val = np.mean(val_loss_list_tmp)
                        min_val_epoch = epoch
                        torch.save(model,'model/yelp/min_val_model')
                    torch.save(model,'model/yelp/model')
    elif method=='eval':
        config = {
            'w2v_length'    :   30,
            'batch_size' : 64,
        }
        w2v_model = Preprocess(config['w2v_length'])
        val_dataset = YelpDataset('/root/Downloads/yelp/kaggle/business_review_covid_val.csv',w2v_model)
        #model = YelpModelv1(val_dataset.embedding,val_dataset.INT_COLS_DIM,text_dim=len(val_dataset.TEXT_COLS)).to(device)
        model = torch.load('model/yelp/min_val_model')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=0)
        model.eval()
        criterion = torch.nn.L1Loss()
        val_loss_list = []
        pred_list = []
        target_list = []

        for step, data in enumerate(val_loader):
            int_array, class_array, text_embedding_array = data[:3]
            y = data[3]

            int_array = int_array.to(device)
            class_array = class_array.to(device)
            text_embedding_array = text_embedding_array.to(device)
            y = y.to(device)

            pred = model(int_array, class_array, text_embedding_array).squeeze()
            loss = criterion(pred, y)
            pred_list.extend(pred.tolist())
            target_list.extend(y.tolist())
            val_loss_list.append(loss.item())

        print('val loss {0}'.format(np.mean(val_loss_list)))
        save_loss(val_loss_list,file='model/yelp/loss_eval.png')
        np.save('model/yelp/loss_eval.npy',np.array(val_loss_list))
        np.save('model/yelp/pred.npy',np.array(pred_list))
        np.save('model/yelp/target.npy',np.array(target_list))

    print('end')