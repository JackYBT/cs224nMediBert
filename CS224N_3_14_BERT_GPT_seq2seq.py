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

class CustomDataset(Dataset):
    def __init__(self, df, encoder_tokenizer, decoder_tokenizer, max_len):
            self.encoder_tokenizer = encoder_tokenizer
            self.decoder_tokenizer = decoder_tokenizer
            self.dataframe = df
            self.patient = df["dialogue"].apply(lambda x: x[0]).to_frame()
            self.doc = df["dialogue"].apply(lambda x: x[1]).to_frame()
            self.max_len = max_len
            print("len(self.patient)",len(self.patient))
            print("len(self.doc)",len(self.doc))            

    def __len__(self):
        return len(self.patient)

    def __getitem__(self, index):
        # grab patient's utterance
        input = str(self.patient.iloc[index])
        input = " ".join(input.split())

        inputs = self.encoder_tokenizer.encode_plus(
            input,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']
        input_token_type_ids = inputs["token_type_ids"]
        
        # grab doc's utterance
        output = str(self.doc.iloc[index])
        output = " ".join(output.split())

        outputs = self.decoder_tokenizer.encode_plus(
            output,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        output_ids = outputs['input_ids'] # of len=200
        output_mask = outputs['attention_mask']
        output_token_type_ids = outputs["token_type_ids"]


        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'input_mask': torch.tensor(input_mask, dtype=torch.long),
            'input_token_type_ids': torch.tensor(input_token_type_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'output_mask': torch.tensor(output_mask, dtype=torch.long),
            'output_token_type_ids': torch.tensor(output_token_type_ids, dtype=torch.long),
        }

class CustomModel(torch.nn.Module):
  def __init__(self, bert_checkpoint, gpt_checkpoint, num_labels, encoder_tokenizer, decoder_tokenizer, temperature=0.5, dropout_rate = 0.1): 
    super(CustomModel,self).__init__() 
    self.num_labels = num_labels 
    self.temperature = temperature
    self.dropout_rate = dropout_rate
    self.encoder_tokenizer = encoder_tokenizer
    self.decoder_tokenizer = decoder_tokenizer

    #Load Model with given checkpoint and extract its body
    myConfig = AutoConfig.from_pretrained(bert_checkpoint,output_hidden_states=True, output_attention=True, temperature=self.temperature)

    self.encoder = AutoModel.from_pretrained(bert_checkpoint,config=myConfig)
    self.decoder = GPT2LMHeadModel.from_pretrained(gpt_checkpoint, add_cross_attention=True).to(device)

    self.dropout = torch.nn.Dropout(self.dropout_rate) 
    self.classifier = torch.nn.Linear(self.encoder.config.hidden_size,num_labels) # load and initialize weights
    self.criterion = torch.nn.CrossEntropyLoss() # define loss function

  def forward(self, encoder_input_ids=None, encoder_attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,labels=None):
    #Extract outputs from the body

    outputs = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, output_hidden_states=True)
    # select the 12th layer 
    hidden_states = outputs.hidden_states[-1]

    # this decoder outputs a dict of loss, logits, past_key_values, hidden_states, attentions, cross_attentions
    seqoutputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states = hidden_states, labels=decoder_input_ids) 
    
    return seqoutputs
  
  def generate(self, encoder_input_ids=None, encoder_attention_mask=None, decoder_input_ids=None):
    print("--- calling generate----")
    outputs = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    # for generator_output, could also pass in encoder_input_ids tokenized by decoder_tokenizer 
    print('decoder_input_ids. to text', self.decoder_tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True, clean_up_tokenization_space=True))
    print('encoder_input_ids to text', self.encoder_tokenizer.batch_decode(encoder_input_ids))
    generator_output = self.decoder.generate(#input_ids=torch.cat((encoder_input_ids, decoder_input_ids), 1),
                                            input_ids = decoder_input_ids,
                                            #token_type_ids=decoder_token_type_ids,
                                           # encoder_attention_mask=encoder_attention_mask,
                                            encoder_hidden_states=hidden_states,
                                           # labels=decoder_input_ids,
                                           # bos_token_id=random.randint(1,30000),
                                            do_sample=True,   
                                            top_k=50, 
                                            max_length = 512,
                                            top_p=0.95
                                            #num_return_sequences=1
    )
    
    print('inside generate, generator_output.shape', generator_output.shape)
    return generator_output

def processDataset(filepath, encoder_tokenizer, decoder_tokenizer, max_len, batch_size):
    # this function returns train_df, test_df, train_dataloader, test_dataloader

    ## Process data
    f = open(filepath)
    data = json.load(f)
    f.close()
    df = pd.DataFrame(data)

    ## Creating the dataset and dataloader for the neural network
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)

    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(len(data)))
    print("TRAIN Dataset: {}".format(len(train_dataset)))
    print("TEST Dataset: {}".format(len(test_dataset)))

    training_set = CustomDataset(train_dataset, encoder_tokenizer, decoder_tokenizer, max_len=max_len)
    testing_set = CustomDataset(test_dataset, encoder_tokenizer, decoder_tokenizer, max_len=max_len)

    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 2
                    }

    test_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 2
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader


def train(optimizer, model, training_loader, testing_loader, device, num_epochs, learning_rate = 0.1):
#------------------------START TRAINING-------------------
    update_count = 0

    start = time.time()
    print('start training....')
    for epoch in range(num_epochs):
        #------------------------training------------------------
        model.train()
        total_losses = 0
        seq_losses = 0
        times = 0
        for _, data in enumerate(training_loader, 0):
            encoder_input = data['input_ids'].to(device, dtype = torch.long)
            mask_encoder_input = data['input_mask'].to(device, dtype = torch.long)
            #encoder_token_type_ids = data['input_token_type_ids'].to(device, dtype = torch.long)
            decoder_input = data['output_ids'].to(device, dtype = torch.long)
            mask_decoder_input = data['output_mask'].to(device, dtype=torch.long)
            #decoder_token_type_ids = data['output_token_type_ids'].to(device, dtype=torch.long)

            outputs= model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
            
            seq_loss = outputs.loss
            seq_loss.backward()

            total_losses += seq_loss.item()
            times += 1
            
            update_count += 1

            optimizer.step()
            optimizer.zero_grad()

            if update_count % 1000 == 0:
                print('-'*20 + f'epoch {epoch}' + '-'*20)
                print(f'update count: {update_count}')
                print(f'total loss: {total_losses / times}')
                print(f'seq loss: {seq_loss}')
                print('-'*20)
                times = 0
                total_losses = 0
        end = time.time()
        print('-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'time: {(end - start)}')
        print(f'total loss: {total_losses / times}')
        start = end

        torch.save(model.state_dict(), "{PATH_NAME}/weights/3_15_bert_gpt_validate_data_testingloader_seq2seq_{EPOCH}.bin".format(PATH_NAME=PATH_NAME, EPOCH=epoch))

def generate_res(model, batch_data, decoder_tokenizer, decoder_prompt, hidden_states=None):
    # make sure your model is on GPU
    device = torch.device('cuda')

    #------------------------LOAD MODEL-----------------
    model.eval()

    #------------------------END LOAD VALIDATE DATA--------------

    generated_sentences = {}
    ind = 0
    #------------------------START SAMPLE GENERETE-------------------
    with torch.no_grad():
        outputs= model.generate(encoder_input_ids = batch_data['input_ids'].to(device, dtype=torch.long),
                                decoder_input_ids=decoder_prompt,
                                encoder_attention_mask=batch_data['input_mask'].to(device, dtype=torch.long)
                                #hidden_states=hidden_states
                                #decoder_attention_mask=decoder_attention_mask
                                )
        # TODO: Check size of outputs in order to determine if the chinese_text is being properly decoded

        for i, sample_output in enumerate(outputs):
            chinese_text = decoder_tokenizer.decode(sample_output, skip_special_tokens=True, clean_up_tokenization_space=True)
            generated_sentences[ind] = [chinese_text]
            assert(len(generated_sentences[ind]) == 1)
            ind += 1
    return generated_sentences

def generate_gts(batch_data, tokenizer):
    # takes in a pandas df with patient's question and doctor's answer 
    # returns a dictionary of index -> doctor's groundtruth answer 
    gts = {}
    encoder_input = batch_data['output_ids'].to(device, dtype = torch.long)
    answers = tokenizer.batch_decode(encoder_input, skip_special_tokens=True, clean_up_tokenization_space=True)
    gts = {i:[answer] for i, answer in enumerate(answers)}
    return gts


def eval_prep(model, batch_data, hidden_states):
    # generate two dictionaries, with ind -> tokenized_gts and ind -> tokenized_pred
    print('---- generating gts ------')
    gts = generate_gts(batch_data, model.decoder_tokenizer)
    print(gts)
    print(' ---- generating res -----')
    decoder_prompt = torch.tensor([model.decoder_tokenizer.encode("医生：", 
            add_special_tokens=True,
            max_length=200,
            pad_to_max_length=True)] * 8).to(device, dtype=torch.long)
    print('decoder_prompt.shape', decoder_prompt.shape)
    res = generate_res(model, batch_data, model.decoder_tokenizer, decoder_prompt, hidden_states)
    print('len(res)', len(res))
    print(res)
    return gts, res

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
    model=CustomModel(bert_checkpoint=bert_checkpoint, gpt_checkpoint=gpt_checkpoint, num_labels=10,encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer).to(device)
    state_dict = torch.load(saved_model_pth, map_location='cuda')
    model.load_state_dict(state_dict)
    print(' ----- finished loading model -----')

    ## evaluate perplexity
    print('------ evaluating ------')
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