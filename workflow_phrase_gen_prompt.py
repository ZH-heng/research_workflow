import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, MvpForConditionalGeneration
from tqdm import tqdm
from torchmetrics.functional.text.rouge import rouge_score
import pandas as pd
import json
import sys
from loguru import logger


class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        inputs = tokenizer(row.content.lower(), padding='max_length', max_length=cfg.max_len_content, truncation=True, return_tensors='pt', add_special_tokens=True)
        y = tokenizer(row.title.lower(), padding='max_length', max_length=cfg.max_len_title, truncation=True, return_tensors='pt')
        # The title of the above text is
        # The title describing research workflow of the above text is
        # Summarize a title for the above text
        # Summarize a title describing research workflow for the above text
        # Summarize the research workflow described in the above text
        # Summarize a research workflow from the above text
        # Summarize the above text with a phrase
        # Generate a title for the above text
        # Generate a title describing research workflow for the above text
        # Generate a research workflow from the above text
        prompt = "Generate a title describing research workflow for the above text: "
        prompt_inputs = tokenizer(prompt.lower(), return_tensors='pt')
        return dict(
            input_ids = torch.cat([inputs['input_ids'].squeeze(0), prompt_inputs['input_ids'].squeeze(0)]),
            attention_mask = torch.cat([inputs['attention_mask'].squeeze(0), prompt_inputs['attention_mask'].squeeze(0)]),
            y_ids = y['input_ids'].squeeze(0)
        )

def predict(dataloader, titles, model):
    model.eval()
    loop = tqdm(dataloader, desc='Predict...')
    preds = []
    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=cfg.max_len_title)
            preds.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    titles = [_.lower() for _ in titles]
    s = rouge_score(preds, titles, use_stemmer=True)
    res = {k:s[k].item() for k in s}
    return res

def make_dataloader():
    df = pd.read_csv(cfg.data_file)
    train_df, valid_df, test_df = df[df.type=='train'], df[df.type=='valid'], df[df.type=='test']
    logger.info(f'train size:{len(train_df)}, valid size:{len(valid_df)}, test size:{len(test_df)}')
    train_set = MyDataset(train_df)
    torch.manual_seed(cfg.seed)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.batch_size)
    valid_set = MyDataset(valid_df)
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=cfg.batch_size_pred)
    test_set = MyDataset(test_df)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=cfg.batch_size_pred)
    return train_loader, valid_loader, test_loader, valid_df['title'].tolist(), test_df['title'].tolist()

def run(model):
    train_loader, valid_loader, test_loader, valid_titles, test_titles = make_dataloader()
    logger.info('start training...')
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
    rouge1_ = 0
    for epoch in range(cfg.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{cfg.epochs}]')
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y_ids = batch['y_ids'].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=y_ids).loss
            loop.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        res = predict(valid_loader, valid_titles, model)
        logger.info(f'epoch:{epoch + 1}, train_loss:{loss.item():.5f}, valid rouge 1/2/L:{res["rouge1_fmeasure"]*100:.2f}/{res["rouge2_fmeasure"]*100:.2f}/{res["rougeL_fmeasure"]*100:.2f}')
        # Save the model when the ROUGE-1 score on the validation set improves.
        if res["rouge1_fmeasure"]>rouge1_:
            rouge1_ = res["rouge1_fmeasure"]
            torch.save(model.state_dict(), cfg.model_save_path)
    # Load the model with the highest ROUGE-1 score on the validation set, and then make predictions on the test set.
    model.load_state_dict(torch.load(cfg.model_save_path))
    res = predict(test_loader, test_titles, model)
    logger.info(f'test eval:{str(res)}')
    logger.info(f'test rouge 1/2/L:{res["rouge1_fmeasure"]*100:.2f}/{res["rouge2_fmeasure"]*100:.2f}/{res["rougeL_fmeasure"]*100:.2f}')


class Config:
    def __init__(self, **kws):
        self.prompt = 9
        self.max_len_title = 22
        self.max_len_content = 316
        self.batch_size = 6
        self.batch_size_pred = 32
        self.seed = 42
        self.lr = 2e-5
        self.epochs = 5
        self.data_file = './data/gen-data.csv'
        self.model_path = './models/flan-t5-large'
        self.model_save_path = './001.pth'
        self.log_path = './log.txt'
        self.__dict__.update(kws)

# This study employed each model’s respective tokenizer to process the dataset. Token length statistics were subsequently computed, 
# and the 90th percentile token length was designated as the max_len. In the phrase generation task, the max_len includes the length of both paragraphs and titles.
# t5: 316 22
# flan-t5 316 22
# bart: 276 19
# mvp: 276  19
# pegasus: 268 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()
logger.remove(handler_id=None)
logger.add(sys.stderr, format='{time:YYYY-MM-DD HH:mm:ss} | {message}')
logger.add(cfg.log_path, encoding='utf-8', format='{time:YYYY-MM-DD HH:mm:ss} | {message}')

if __name__=="__main__":

    cfg = Config()
    logger.info(json.dumps(cfg.__dict__, indent=4, ensure_ascii=False))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(cfg.model_path)
    run(model)
