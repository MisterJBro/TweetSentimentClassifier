from pytorch_lightning import LightningModule
from models import ModelBert, ModelRoberta, ModelXLNet, ModelDistilBert, ModelAlbert, ModelBertweet, ModelTwitterRoberta
import os
from collections import Counter
from dataset import TweetDataset
import torch
from torch.utils.data import DataLoader
from sklearn import metrics

# Ensemble model, loads all respective models and performs a majority vote for the given input strings
class Ensemble(LightningModule):
    def __init__(self, models=['bert', 'roberta', 'xlnet', 'distilbert', 'albert', 'bertweet', 'twitterroberta']):
        super(Ensemble, self).__init__()
        self.models = []
        if 'bert' in models:
            path = self.get_model_path('bert')
            if path is not None:
                self.bert = ModelBert.load_from_checkpoint(path)
                self.models.append(self.bert)
                print('Loaded bert model')
            else:
                print('No bert model available in checkpoints! Skipping bert model for ensemble')
        if 'roberta' in models:
            path = self.get_model_path('roberta')
            if path is not None:
                self.roberta = ModelRoberta.load_from_checkpoint(path)
                self.models.append(self.roberta)
                print('Loaded roberta model')
            else:
                print('No roberta model available in checkpoints! Skipping roberta model for ensemble')
        if 'xlnet' in models:
            path = self.get_model_path('xlnet')
            if path is not None:
                self.xlnet = ModelXLNet.load_from_checkpoint(path)
                self.models.append(self.xlnet)
                print('Loaded xlnet model')
            else:
                print('No xlnet model available in checkpoints! Skipping xlnet model for ensemble')
        if 'distilbert' in models:
            path = self.get_model_path('distilbert')
            if path is not None:
                self.distilbert = ModelDistilBert.load_from_checkpoint(path)
                self.models.append(self.distilbert)
                print('Loaded distilbert model')
            else:
                print('No distilbert model available in checkpoints! Skipping distilbert model for ensemble')
        if 'albert' in models:
            path = self.get_model_path('albert')
            if path is not None:
                self.albert = ModelAlbert.load_from_checkpoint(path)
                self.models.append(self.albert)
                print('Loaded albert model')
            else:
                print('No albert model available in checkpoints! Skipping albert model for ensemble')
        if 'bertweet' in models:
            path = self.get_model_path('bertweet')
            if path is not None:
                self.bertweet = ModelBertweet.load_from_checkpoint(path)
                self.models.append(self.bertweet)
                print('Loaded bertweet model')
            else:
                print('No bertweet model available in checkpoints! Skipping bertweet model for ensemble')
        if 'twitterroberta' in models:
            path = self.get_model_path('twitterroberta')
            if path is not None:
                self.twitterroberta = ModelTwitterRoberta.load_from_checkpoint(path)
                self.models.append(self.twitterroberta)
                print('Loaded twitterroberta model')
            else:
                print('No twitterroberta model available in checkpoints! Skipping twitterroberta model for ensemble')

        for m in self.models:
            m.eval()
            m.freeze()

    def get_model_path(self, model_name):
        paths = [f for f in os.listdir('../checkpoints') if f.lower().startswith(model_name.lower() + '_epoch')]

        if len(paths) == 0:
            return None
        if len(paths) == 1:
            return '../checkpoints/' + paths[0]

        # If more then one path exists, use the one with highest val score
        scores = [float(x.split('=')[-1][:-5]) for x in paths]
        index, _ = max(enumerate(scores), key=lambda x: x[1])

        return '../checkpoints/' + paths[index]

    def forward(self, input, output_uncertainty=False):
        preds = []
        for model in self.models:
            tokens = model.tokenize(input)
            input_ids = tokens['input_ids'].to(self.device)
            att_mask = tokens['attention_mask'].to(self.device)
            pred = model(input_ids, att_mask).argmax(dim=1).detach().cpu().item()
            preds.append(pred)
        # Get the class with highest votes. In case of a tie, pick in this order: label 2, label 1, label 0.
        # That is based on their portion within the datasets
        pred = Counter(preds)
        print(pred)
        max_votes = 0
        max_labels = []
        for key, val in pred.items():
            if val > max_votes:
                max_labels = [int(key)]
                max_votes = val
            elif val == max_votes:
                max_labels.append(int(key))

        pred = max(max_labels)
        out = [pred]
        if output_uncertainty:
            out.append(max_votes / len(self.models))
        return out

    def test_step(self, batch, batch_idx):
        input, label = batch

        att_mask = input['attention_mask'].squeeze(1)
        input_id = input['input_ids'].squeeze(1)

        output = self(input_id, att_mask)
        loss = self.criterion(output, label)

        pred = torch.argmax(output, dim=1)
        self.accuracy(pred, label)
        f1 = metrics.f1_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='micro')

        #self.log('val_loss', loss, prog_bar=True)
        #self.log('val_acc', self.accuracy, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        return {'loss': loss, 'pred': pred}

    def test_epoch_end(self, outs):
        preds = torch.cat([o['pred'] for o in outs], 0)
        self.log('val_0', (preds==0).sum().item()/len(outs), prog_bar=True)
        self.log('val_1', (preds==1).sum().item()/len(outs), prog_bar=True)
        self.log('val_2', (preds==2).sum().item()/len(outs), prog_bar=True)

    def setup(self, stage=None):
        self.test_ds = TweetDataset('../datasets/preprocessed/dev.csv', self.tokenize)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=1)

