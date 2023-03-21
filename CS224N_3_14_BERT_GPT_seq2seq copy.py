# pip install pycocoevalcap
# import cocoeval package
from evaluation.bleu.bleu import Bleu
from evaluation.rouge import Rouge
from torch import cuda, nn
import time
import json
import random
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
from tqdm.auto import tqdm
from util import sequence_cross_entropy_with_logits
import tqdm
# from perplexity import calculate_perplexity
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig, BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, GPT2Config, AdamW
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoConfig

from CS224N_3_16_BERT_GPT_multitask import *
PATH_NAME = "./"
device = 'cuda' if cuda.is_available() else 'cpu'
pd.options.display.max_colwidth = 1000

def main():
    ## Sections of config
    MAX_LEN = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-05

    bert_checkpoint = "trueto/medbert-base-wwm-chinese"
    encoder_tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint,  model_max_len=512)
    EPOCHS=3
    saved_model_pth = "/home/ubuntu/CS_224N/weights/Multitask/3_17_bert_gpt_multitask_allpatient_1.bin"
    json_pth = "/home/ubuntu/CS_224N/Dataset/patient_data.json"
    gpt_checkpoint = "uer/gpt2-chinese-cluecorpussmall"
    decoder_tokenizer = AutoTokenizer.from_pretrained(gpt_checkpoint)
    warmup_steps = 1e2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #train_df, test_df, training_loader, testing_loader = processDataset(json_pth, encoder_tokenizer, decoder_tokenizer, MAX_LEN, BATCH_SIZE)
    training_loader, testing_loader = processDataset(json_pth, encoder_tokenizer, decoder_tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
    ## load model
    print(' ----- loading model ------')
    model=CustomModel(checkpoint=bert_checkpoint, gpt_checkpoint=gpt_checkpoint, num_labels=10,encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer).to(device)
    state_dict = torch.load(saved_model_pth, map_location='cuda')
    model.load_state_dict(state_dict)
    print(' ----- finished loading model -----')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    ## evaluate perplexity
    print('------ evaluating ------')
    train(optimizer, model, training_loader, testing_loader, device, EPOCHS, LEARNING_RATE)

    # passing in the testing set, get perplexity values on training
    # the below function takes in gts and res, both dict()
    # they should have the same keys from id -> text
    for _, data in enumerate(training_loader, 0):
        gts, res = eval_prep(model, data, hidden_states=None)
        blue_class = Bleu()
        rouge_class = Rouge()
        bleu_score, bleu_info = blue_class.compute_score(gts, res)
        print('blue score is ', bleu_score)
        rouge_avg, rouge_array = rouge_class.compute_score(gts, res)
        print('rouge average score is ', rouge_avg)

if __name__ == '__main__':
    main()