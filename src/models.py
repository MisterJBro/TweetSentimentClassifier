# All NLP classification models for the Ensemble
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, AlbertModel, AlbertTokenizer, AutoModel, AutoTokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import TweetDataset
from transformers import logging as transformer_logging
transformer_logging.set_verbosity_error()
import logging
#logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
from sklearn import metrics
from transformers import AdamW, XLNetTokenizer, XLNetModel

# Base class for all Ensemble models
class ModelBase(LightningModule):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.accuracy = Accuracy()
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        input, label = batch

        att_mask = input['attention_mask'].squeeze(1)
        input_id = input['input_ids'].squeeze(1)

        output = self(input_id, att_mask)
        loss = self.criterion(output, label)

        self.log('train_loss', loss, on_epoch=True, logger=True)
        return loss

    def training_epoch_end(self, outs):
        print('')

    def get_metrics(self):
        # don't show the version number
        items = super().get_metrics()
        items.pop("v_num", None)
        return items

    def validation_step(self, batch, batch_idx):
        input, label = batch

        att_mask = input['attention_mask'].squeeze(1)
        input_id = input['input_ids'].squeeze(1)

        output = self(input_id, att_mask)
        loss = self.criterion(output, label)

        pred = torch.argmax(output, dim=1)
        self.accuracy(pred, label)
        #self.log('val_loss', loss, prog_bar=True)
        #self.log('val_acc', self.accuracy, prog_bar=True)

        return {'loss': loss, 'pred': pred, 'label': label}

    def validation_epoch_end(self, outs):
        preds = torch.cat([o['pred'].reshape(-1) for o in outs], 0)
        labels = torch.cat([o['label'].reshape(-1) for o in outs], 0)
        f1 = metrics.f1_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_0', (preds==0).sum().item()/(len(outs)*self.batch_size), prog_bar=True)
        self.log('val_1', (preds==1).sum().item()/(len(outs)*self.batch_size), prog_bar=True)
        self.log('val_2', (preds==2).sum().item()/(len(outs)*self.batch_size), prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def setup(self, stage=None):
        #if stage == 'fit' or stage is None:
        # Get datasets with dataloaders
        self.train_ds = TweetDataset('../datasets/preprocessed/train.csv', self.tokenize)
        self.dev_ds = TweetDataset('../datasets/preprocessed/dev.csv', self.tokenize)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.dev_ds, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.dev_ds, batch_size=self.batch_size, num_workers=1)

    def tokenize(self, s):
        return self.tokenizer(s, add_special_tokens=True, padding='max_length', max_length = 512, truncation=True, return_tensors='pt')


class ModelBert(ModelBase):
    def __init__(self, dropout=0.3, num_classes=3, lr=2e-5, batch_size=16):
        super(ModelBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.save_hyperparameters()

    def forward(self, input_id, att_mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=att_mask, return_dict=False)
        out = self.dropout(pooled_output)
        out =  self.linear(out)

        return out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class ModelRoberta(ModelBase):
    def __init__(self, dropout=0.3, num_classes=3, lr=2e-5, batch_size=16):
        super(ModelRoberta, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.roberta.config.hidden_size, num_classes)
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        self.save_hyperparameters()

    def forward(self, input_id, att_mask):
        _, pooled_output = self.roberta(input_ids=input_id, attention_mask=att_mask, return_dict=False)
        out = self.dropout(pooled_output)
        out = self.linear(out)

        return out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class ModelXLNet(ModelBase):
    def __init__(self, dropout=0.3, num_classes=3, lr=2e-5, batch_size=16):
        super(ModelXLNet, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        #self.xlnet = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",num_labels=num_classes)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.xlnet.config.hidden_size, num_classes)
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        self.save_hyperparameters()

    def forward(self, input_id, att_mask):
        out = self.xlnet(input_ids=input_id, attention_mask=att_mask, return_dict=False)
        out = out[0].mean(1)
        out = self.dropout(out)
        out = self.linear(out)

        # out = self.xlnet(input_ids=input_id, attention_mask=att_mask)

        return out

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class ModelDistilBert(ModelBase):
    def __init__(self, dropout=0.3, num_classes=3, lr=2e-5, batch_size=16):
        super(ModelDistilBert, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        self.save_hyperparameters()

    def forward(self, input_id, att_mask):
        out = self.distilbert(input_ids=input_id, attention_mask=att_mask, return_dict=False)
        out = out[0][:, 0]
        out = self.dropout(out)
        out = self.linear(out)

        return out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class ModelAlbert(ModelBase):
    def __init__(self, dropout=0.3, num_classes=3, lr=2e-5, batch_size=16):
        super(ModelAlbert, self).__init__()
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.albert.config.hidden_size, num_classes)
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.save_hyperparameters()

    def forward(self, input_id, att_mask):
        _, pooled_output = self.albert(input_ids=input_id, attention_mask=att_mask, return_dict=False)
        out = self.dropout(pooled_output)
        out = self.linear(out)

        return out

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class ModelBertweet(ModelBase):
    def __init__(self, dropout=0.3, num_classes=3, lr=2e-5, batch_size=16):
        super(ModelBertweet, self).__init__()
        self.bertweet = AutoModel.from_pretrained('vinai/bertweet-large')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bertweet.config.hidden_size, num_classes)
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-large', use_fast=False)
        self.save_hyperparameters()

    def forward(self, input_id, att_mask):
        _, pooled_output = self.bertweet(input_ids=input_id, attention_mask=att_mask, return_dict=False)
        out = self.dropout(pooled_output)
        out = self.linear(out)

        return out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def tokenize(self, s):
        return self.tokenizer(s, add_special_tokens=True, padding='max_length', max_length=312, truncation=True, return_tensors='pt')


class ModelTwitterRoberta(ModelBase):
    def __init__(self, dropout=0.3, num_classes=3, lr=2e-5, batch_size=16):
        super(ModelTwitterRoberta, self).__init__()
        self.twitterroberta = AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.twitterroberta.config.hidden_size, num_classes)
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        self.save_hyperparameters()

    def forward(self, input_id, att_mask):
        _, pooled_output = self.twitterroberta(input_ids=input_id, attention_mask=att_mask, return_dict=False)
        out = self.dropout(pooled_output)
        out = self.linear(out)

        return out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
