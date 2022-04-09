import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from gensim.models import word2vec
import re
import torch

from yelp import YelpModelv0
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
    INT_COLS = ['review_count','stars_x','stars_y','useful']
    TEXT_COLS = ['categories','text']
    COVID_COLS = ['highlights', 'delivery or takeout', 'Grubhub enabled', 'Call To Action enabled', 'Request a Quote Enabled', 'Covid Banner', 'Virtual Services Offered']

    def __init__(self,data_path,w2v:Preprocess=None,maxlen=100):
        self.raw_data = pd_load_csv(data_path,cols=self.USECOLS).iloc[:int(maxlen)]
        self.w2v = w2v
        self.data = {}
        self.word2vec_dataset = []

        if w2v:
            self.embedding = w2v.make_embedding()
        else:
            self.embedding = None

        self.init_dataset()

    def init_dataset(self):
        for i in range(len(self)):
            int_array = []
            class_array = []
            text_embedding_array = []
            covid_array = []

            row = self.raw_data.iloc[i]

            for col in self.INT_COLS:
                int_array.append(int(row[col]))

            for col in self.CLASS_COLS:
                class_array.append(int(row[col]))

            for col in self.TEXT_COLS:
                sentence = self.sentence_split(row[col])
                sentence_embedding = self.w2v.sentence_word2idx(sentence)
                text_embedding_array.append(sentence_embedding)

            for col in self.COVID_COLS:
                col_value = row[col]
                col_value = str(col_value).upper() == 'FALSE'
                covid_result = 0 if col_value else 1
                covid_array.append(covid_result)
            covid_target = np.sum(covid_array)

            int_array = torch.FloatTensor(int_array)
            class_array = torch.IntTensor(class_array)
            text_embedding_array = torch.stack(text_embedding_array, dim=0)

            self.data[i] = (int_array, class_array, text_embedding_array, covid_target)

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
    METHODS = ['train_w2v', 'inspect_dataset','inspect_model']
    method = METHODS[2]
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
        dataset = YelpDataset('/root/Downloads/yelp/kaggle/business_review_covid.csv',w2v_model)
        model = YelpModelv0(dataset.embedding,len(dataset.INT_COLS))
        train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=32,shuffle=False,num_workers=0)
        for step, data in enumerate(train_loader):
            x=data[:3]
            y=data[3]
            model(x)
    print('end')