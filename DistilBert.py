from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import numpy as np

tokenizer_global = None

class DistilBert():
    def __init__(self,
                 n_classes=3,
                 multilabel_or_multiclass="multiclass",
                 is_cased = False,
                 TRAIN_BATCH_SIZE = 8,
                 VALID_BATCH_SIZE = 8,
                 EPOCHS = 1,
                 LEARNING_RATE = 3e-05
                 ):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.MAX_LEN = 512
        if not is_cased:
            self.model_name = "distilbert-base-uncased"
        else:
            self.model_name = "distilbert-base-cased"
        self.n_classes = n_classes
        self.multilabel_or_multiclass = multilabel_or_multiclass
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.VALID_BATCH_SIZE = VALID_BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.model = ModelClass(n_classes = n_classes, model_name=self.model_name, multilabel_or_multiclass = multilabel_or_multiclass)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr=LEARNING_RATE)
        if self.multilabel_or_multiclass == "multilabel":
            self.loss = nn.BCELoss()
            self.collate_fn = collate_fn_multilabel
        elif self.multilabel_or_multiclass == "multiclass":
            self.loss = nn.CrossEntropyLoss()
            self.collate_fn = collate_fn_multiclass
        else:
            print("ERROR ....Wrong Task......")

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        training_set = dataset(texts, labels, tokenizer=self.model.tokenizer, max_len=self.MAX_LEN)
        train_params = {
            "batch_size" : self.TRAIN_BATCH_SIZE,
            "shuffle" : True,
            "num_workers" : 0,
            "collate_fn" : self.collate_fn
        }
        training_loader = DataLoader(training_set, **train_params)

        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_set = dataset(val_texts, val_labels, tokenizer=self.model.tokenizer, max_len=self.MAX_LEN)
            val_params = {
                "batch_size" : self.VALID_BATCH_SIZE,
                "shuffle" : False,
                "num_workers" : 0,
                "collate_fn" : self.collate_fn 
            }
            val_loader = DataLoader(val_set, **val_params)

        for epoch in range(self.EPOCHS):
            self.train(epoch, training_loader, val_loader, training_set.len_)


    def train(self, epoch, training_loader, val_loader, train_set_len):
        prev = 0
        for i , (batch, labels) in enumerate(training_loader):
            self.model.train()
            ids = batch['input_ids'].to(self.device)
            masks = batch['attention_mask'].to(self.device)
            labels = labels.to(self.device)
            output = self.model(input_ids = ids, attention_mask = masks)
            loss_val = self.loss(output, labels)

            if int((i*100*self.TRAIN_BATCH_SIZE)/train_set_len) > prev:
                print("Evaluating ...")
                validation_loss = 0.0
                cnt = 0
                print("Epoch", epoch, "step_percentage", int((i*100*self.TRAIN_BATCH_SIZE)/train_set_len) ,"%")
                if val_loader is not None:
                    self.model.eval()
                    for j, (val_batch, val_labels) in enumerate(val_loader):
                        val_ids = val_batch['input_ids'].to(self.device)
                        val_masks = val_batch['attention_mask'].to(self.device)
                        val_labels = val_labels.to(self.device)
                        val_loss = self.loss(self.model(val_ids, val_masks), val_labels)
                        validation_loss += val_loss.item()
                        cnt+=1
                    validation_loss /= cnt
                    print("Train loss", loss_val.item(),"Validation loss", validation_loss)
                else:
                    print("Train loss", loss_val.item())
                prev = int((i*100*self.TRAIN_BATCH_SIZE)/train_set_len)

            loss_val.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict(self, texts):
        self.model.eval()
        prediction_set = dataset(texts, None, tokenizer=self.model.tokenizer, max_len=self.MAX_LEN)
        prediction_params = {
            "batch_size" : self.VALID_BATCH_SIZE,
            "shuffle" : False,
            "num_workers" : 0,
            "collate_fn" : self.collate_fn 
        }
        prediction_loader = DataLoader(prediction_set, **prediction_params)

        probs = []
        for i, batch in enumerate(prediction_loader):
            pred_ids = batch['input_ids'].to(self.device)
            pred_masks = batch['attention_mask'].to(self.device)
            probabilities = nn.functional.softmax(self.model(pred_ids, pred_masks), dim=1).cpu().detach().numpy()
            if self.multilabel_or_multiclass == "multilabel":
                probs.append(probabilities[0])
            elif self.multilabel_or_multiclass == "multiclass":
                probs.append(probabilities[0])

        return np.array(probs)


class ModelClass(nn.Module):
    def __init__(self, n_classes = 3, model_name = None, multilabel_or_multiclass = "multiclass", **kwargs):
        super(ModelClass, self).__init__(**kwargs)
        if multilabel_or_multiclass not in ["multilabel", "multiclass"]:
            print("Error ....... Making it multiclass")
            multilabel_or_multiclass = "multiclass"
        
        global tokenizer_global
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.multilabel_or_multiclass = multilabel_or_multiclass
        self.final_layer = nn.Linear(768, n_classes)
        tokenizer_global = self.tokenizer

    def forward(self, input_ids, attention_mask):
        layer_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:,0]
        if self.multilabel_or_multiclass == "multilabel":
            layer_2 = torch.sigmoid(self.final_layer(layer_1))
        else:
            layer_2 = self.final_layer(layer_1)
        return layer_2


def collate_fn_multilabel(batch):
    texts = [t['text'] for t in batch]
    labels = [t['label'] for t in batch]
    batch = tokenizer_global.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, return_tensors='pt')
    if labels[0] is not None:
        labels = torch.FloatTensor(labels)
        return batch, labels
    return batch

def collate_fn_multiclass(batch):
    texts = [t['text'] for t in batch]
    labels = [t['label'] for t in batch]
    batch = tokenizer_global.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, return_tensors='pt')
    if labels[0] is not None:
        labels = torch.LongTensor(labels)
        return batch, labels
    return batch

class dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.len_ = len(texts)
        self.data = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
      
    def __getitem__(self, index):
        text = str(self.data[index])
        text = " ".join(text.split())
        if self.labels is not None:
            label = self.labels[index]
            return {"text":text, "label":label}
        return {"text":text, "label":None}

    def __len__(self):
        return self.len_
