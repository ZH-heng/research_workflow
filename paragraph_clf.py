import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer, AutoConfig, RobertaModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
from loguru import logger


class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        inputs = tokenizer(row.content.lower(), padding='max_length', max_length=cfg.max_len, truncation=True, return_tensors='pt')
        return dict(
            input_ids = inputs['input_ids'].squeeze(0),
            attention_mask = inputs['attention_mask'].squeeze(0),
            label = row.tag
        )

class CLFmodel(nn.Module):
    def __init__(self, pretrained_model, hidden_size):
        super(CLFmodel, self).__init__()
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        seq_out = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(seq_out[1])
        logits = self.linear(logits)
        return logits

def predict(model, dataloader):
    model.eval()
    pred_label, y_true = [], []
    with torch.no_grad():
        loop = tqdm(dataloader, total=len(dataloader), desc='Predict...')
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            outputs = outputs.argmax(dim=1)
            pred_label.extend(outputs.detach().cpu().numpy().tolist())
            y_true.extend(batch['label'].numpy().tolist())
    acc = accuracy_score(y_true, pred_label)
    p = precision_score(y_true, pred_label)
    r = recall_score(y_true, pred_label)
    f1 = f1_score(y_true, pred_label)
    return acc, p, r, f1

def make_dataloader():
    df = pd.read_csv(cfg.data_file)
    train_df, valid_df, test_df = df[df.type=="train"], df[df.type=="valid"], df[df.type=="test"]
    logger.info(f'train size:{len(train_df)}, valid size:{len(valid_df)}, test size:{len(test_df)}')
    trainset = MyDataset(train_df)
    torch.manual_seed(cfg.seed)
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True)
    validset = MyDataset(valid_df)
    validloader = DataLoader(validset, batch_size=cfg.batch_size_pred, shuffle=False)
    testset =  MyDataset(test_df)
    testloader = DataLoader(testset, batch_size=cfg.batch_size_pred, shuffle=False)
    return trainloader, validloader, testloader

def run(pretrained_model, hidden_size):
    trainloader, validloader, testloader = make_dataloader()
    model = CLFmodel(pretrained_model, hidden_size)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    acc_ = 0
    for epoch in range(cfg.epochs):
        model.train()
        loop = tqdm(trainloader, desc=f'Epoch [{epoch+1}/{cfg.epochs}]')
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())
        acc, p, r, f1 = predict(model, validloader)
        logger.info(f'epoch:{epoch + 1}, train loss:{loss.item():.4f}, valid acc:{acc*100:.2f}, P/R/F1:{p*100:.2f}/{r*100:.2f}/{f1*100:.2f}')
        # Save the model when the accuracy metric on the validation set improves.
        if acc>acc_:
            acc_ = acc
            torch.save(model.state_dict(), cfg.model_save_path)
    torch.cuda.empty_cache()
    # Load the model with the highest accuracy on the validation set for test set prediction.
    model.load_state_dict(torch.load(cfg.model_save_path))
    acc, p, r, f1 = predict(model, testloader)
    logger.info(f'test acc:{acc*100:.2f}, P/R/F1:{p*100:.2f}/{r*100:.2f}/{f1*100:.2f}')

class Config:
    def __init__(self, **kws):
        self.max_len = 228
        self.batch_size = 64
        self.batch_size_pred = 256
        self.seed = 42
        self.lr = 3e-5
        self.epochs = 5
        self.data_file = './data/clf-data.csv'
        self.model_path = './models/scibert_scivocab_uncased'
        self.model_save_path = "./001.pth"
        self.log_path = "./log.txt"
        self.__dict__.update(kws)

# This study employed each model’s respective tokenizer to process the dataset. Token length statistics were subsequently computed, 
# and the 90th percentile token length was designated as the max_len.
# - scibert_scivocab_uncased  228
# - bert-base-uncased  237
# - roberta-base  232

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
logger.remove(handler_id=None)
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {message}")
logger.add(cfg.log_path, encoding="utf-8", format="{time:YYYY-MM-DD HH:mm:ss} | {message}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
hidden_size = AutoConfig.from_pretrained(cfg.model_path).hidden_size

for i in range(1, 6):
    cfg = Config(lr=i*1e-5)
    logger.info(json.dumps(cfg.__dict__, indent=4, ensure_ascii=False))
    pretrained_model = BertModel.from_pretrained(cfg.model_path)
    run(pretrained_model, hidden_size)

