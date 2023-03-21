from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import json
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import time
import datetime
import random 
import torch
import wandb
from evaluation.bleu.bleu import Bleu
from evaluation.rouge import Rouge
wandb.init(project="CS224N_3_18_GPT")
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, TextGenerationPipeline, GPT2LMHeadModel, AdamW, TrainingArguments, Trainer, AutoModelWithLMHead, DataCollatorForLanguageModeling, pipeline
PATH_NAME = "./weights/GPT"
pd.options.display.max_colwidth = 1000

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.dialogue = dataframe.dialogue
        self.max_len = max_len
        self.patient = dataframe['dialogue'].apply(lambda x: x[0]).to_frame()
        self.doc = dataframe['dialogue'].apply(lambda x: x[1]).to_frame()
        self.dialogue = dataframe['dialogue']

    def __len__(self):
        return len(self.dialogue)

    def __getitem__(self, index):
        input = self.patient.iloc[index]["dialogue"] + " 医生："
        # self.patient.iloc[index] = currentString
        # print("self.patient.iloc[index][patient]", self.patient.iloc[index]["patient"])
        # input = " ".join(input.split())

        inputs = self.tokenizer.encode_plus(
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

        output = self.doc.iloc[index]["dialogue"]


        outputs = self.tokenizer.encode_plus(
            output,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        output_ids = outputs['input_ids']
        output_mask = outputs['attention_mask']
        output_token_type_ids = outputs["token_type_ids"]

        dialogue = "".join(self.dialogue.iloc[index])
        dialogues = self.tokenizer.encode_plus(
            dialogue,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        dialogue_ids = dialogues['input_ids']
        dialogue_mask = dialogues['attention_mask']
        dialogue_token_type_ids = dialogues["token_type_ids"]


        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'input_mask': torch.tensor(input_mask, dtype=torch.long),
            'input_token_type_ids': torch.tensor(input_token_type_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'output_mask': torch.tensor(output_mask, dtype=torch.long),
            'output_token_type_ids': torch.tensor(output_token_type_ids, dtype=torch.long),
            'dialogue_ids': torch.tensor(dialogue_ids, dtype=torch.long),
            'dialogue_mask': torch.tensor(dialogue_mask, dtype=torch.long),
            'dialogue_token_type_ids': torch.tensor(dialogue_token_type_ids, dtype=torch.long)
        }
    
def processDataset(filepath, tokenizer, max_len, batch_size):
    ## Process data
    f = open(filepath)
    data = json.load(f)
    f.close()
    df = pd.DataFrame(data)

    ## Creating the dataset and dataloader for the neural network
    train_size = 0.8

    train_dataset=df.sample(frac=train_size,random_state=200)
    df_remaining = df.drop(train_dataset.index).reset_index(drop=True)
    val_dataset=df_remaining.sample(frac=0.5,random_state=200)
    test_dataset=df_remaining.drop(val_dataset.index).reset_index(drop=True)

    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)


    print("FULL Dataset: {}".format(len(data)))
    print("TRAIN Dataset: {}".format(len(train_dataset)))
    print("VALIDATION Dataset: {}".format(len(val_dataset)))
    print("TEST Dataset: {}".format(len(test_dataset)))

    
    training_set = CustomDataset(train_dataset, tokenizer, max_len=max_len)
    testing_set = CustomDataset(test_dataset, tokenizer, max_len=max_len)
    validation_set = CustomDataset(val_dataset, tokenizer, max_len=max_len)

    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 2
                    }

    test_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 2
                    }
    validation_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 2
                    }


    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    validation_loader = DataLoader(validation_set, **validation_params)
    return training_loader, validation_loader, testing_loader

def generate_gts(iterationIndex, encoder_input, tokenizer):
    # takes in a pandas df with patient's question and doctor's answer 
    # returns a dictionary of index -> doctor's groundtruth answer 
    gts = {}
    encoder_input = encoder_input.to(device, dtype = torch.long)
    answers = tokenizer.batch_decode(encoder_input, skip_special_tokens=True, clean_up_tokenization_space=True)
    
    for i, answer in enumerate(answers):
        curIndex = iterationIndex*8 + i
        gts = {**{curIndex:[answer]}, **gts}
    return gts

def eval_prep(model, decoder_input, iterationIndex, tokenizer, generator, encoder_input, encoder_mask):

    gts = {}
    res = {}

    gts = generate_gts(iterationIndex, decoder_input, tokenizer)

    for index, batch_chinese_text in enumerate(gts):
        chinese_text = generator(tokenizer.decode(encoder_input[index].to('cpu'), pad_token_id=tokenizer.pad_token_id, attention_mask = encoder_mask, skip_special_tokens=True, clean_up_tokenization_space=True), max_length=300, num_return_sequences=1)[0]['generated_text']
        print("input Chinese text: ", tokenizer.decode(encoder_input[index].to('cpu'), pad_token_id=tokenizer.pad_token_id, attention_mask = encoder_mask, skip_special_tokens=True, clean_up_tokenization_space=True))
        print("generated Chinese text: ", chinese_text)
        print("groundtruth Chinese text", tokenizer.decode(decoder_input[index].to('cpu'), pad_token_id=tokenizer.pad_token_id, skip_special_tokens=True, clean_up_tokenization_space=True))
        curIndex = 8*iterationIndex + index
        res[curIndex] = [chinese_text]

        assert(len(res[curIndex]) == 1)
    return gts, res

def validation(model, validation_loader, tokenizer, device):
    final_gts = {}
    final_res = {}
    print("length of validation loader: ", len(validation_loader))
    for curIndex, data in enumerate(validation_loader, 0):
        
        encoder_input = data['input_ids'].to(device, dtype = torch.long)
        encoder_input_mask = data['input_mask'].to(device, dtype = torch.long)
        decoder_input = data['output_ids'].to(device, dtype=torch.long)

        generator = pipeline('text-generation', model=model.to('cpu'), tokenizer=tokenizer)

        gts, res = eval_prep(model, decoder_input, curIndex, tokenizer, generator, encoder_input, encoder_input_mask)
        model.to(device)

        final_gts.update(gts)
        final_res.update(res)

        # From 1-gram to 4-gram, so is 1X4 matrix
        blue_class = Bleu()
        rouge_class = Rouge()
        bleu_score, bleu_info = blue_class.compute_score(final_gts, final_res)
        print(f'Iteration {curIndex} bleu score is {bleu_score}')
        rouge_avg, rouge_array = rouge_class.compute_score(final_gts, final_res)
        print(f'Iteration {curIndex} rouge average score is {rouge_avg}') 
        wandb.log({'bleu_score[0]': bleu_score[0], 'bleu_score[1]': bleu_score[1], 'bleu_score[2]': bleu_score[2], 'bleu_score[3]': bleu_score[3], 'rouge_avg': rouge_avg}) 
 
        if curIndex % 10 == 0:
            print("############################")
            sorted_keys_gts = sorted(gts.keys(), reverse=True)

            for index, key in enumerate(sorted_keys_gts):
                print(f"Input patient symptoms {index}:", tokenizer.decode(encoder_input[index], skip_special_tokens=True, clean_up_tokenization_space=True))
                print(f"gts {key}", gts[key])
                print(f"res {key}", res[key])

            print("############################")
    blue_class = Bleu()
    rouge_class = Rouge()
    bleu_score, bleu_info = blue_class.compute_score(final_gts, final_res)
    print(f'FINAL bleu score is {bleu_score}')
    rouge_avg, rouge_array = rouge_class.compute_score(final_gts, final_res)
    print(f'FINAL rouge average score is {rouge_avg}') 
    print("#"*50)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def GPTtrain(optimizer, model, training_loader, val_loader, tokenizer, device, lr_scheduler, num_epochs, learning_rate = 0.1):
    t0 = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss_current_epoch = 0
        for curIndex,data in enumerate(training_loader, 0):
            model.zero_grad() 
            ids = data['dialogue_ids'].to(device, dtype = torch.long)
            mask = data['dialogue_mask'].to(device, dtype = torch.long)
            token_type_ids = data['dialogue_token_type_ids'].to(device, dtype = torch.long)



            outputs = model(ids, attention_mask=mask, labels=ids, token_type_ids=token_type_ids)

            loss = outputs.loss
            total_loss_current_epoch += loss.item()


            wandb.log({'cumulative Loss Seq2Seq Accuracy': total_loss_current_epoch / (curIndex + 1), 'current_batch_loss': loss})

            # Get sample every x batches.
            if curIndex% 250 == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(curIndex, len(training_loader), loss, elapsed))
                print('cumulative Loss Seq2Seq Accuracy', total_loss_current_epoch / (curIndex + 1), 'current_batch_loss', loss)
                
            # if curIndex% 45000 == 0:  
            #     model.eval()
            #     with torch.no_grad():
            #         validation(model, val_loader, tokenizer, 'cuda')
                
            #     torch.save(model.state_dict(), "{PATH_NAME}/{FILE_NAME}_{EPOCH}".format(PATH_NAME=PATH_NAME, FILE_NAME = '3_19_GPT_gen.bin', EPOCH=epoch))

            #     model.train()

            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        #print(f"Epoch {epoch} - Validation Accuracy: {accuracy}")
        torch.save(model.state_dict(), "{PATH_NAME}/{FILE_NAME}_{EPOCH}".format(PATH_NAME=PATH_NAME, FILE_NAME = '3_19_GPT_gen.bin', EPOCH=epoch))
        epoch_loss = total_loss_current_epoch / len(training_loader)
        wandb.log({'epoch_loss': epoch_loss})
        print(f"Epoch {epoch} - Loss: {epoch_loss}")
        print("#"*50)

def main():
    # configs
    MAX_LEN = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-05
    CUDA_LAUNCH_BLOCKING = 1
    checkpoint = "uer/gpt2-chinese-cluecorpussmall"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_len=512)
    EPOCHS = 10
    epsilon = 1e-8

    warmup_steps = 1e2
    
    #1. preprocess the data
    json_pth = 'Dataset/patient_data.json'
    training_loader, val_loader, testing_loader = processDataset(json_pth, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
    
    #2. define the  model
    model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)
    optimizer = AdamW(model.parameters(),
                  lr = LEARNING_RATE,
                  eps = epsilon
                )
    total_steps = len(training_loader) * EPOCHS
    lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

    # GPTtrain(optimizer, model, training_loader, val_loader, tokenizer, device, lr_scheduler, EPOCHS, LEARNING_RATE)
    validation(model, testing_loader, tokenizer, device)
if __name__ == '__main__':
    main()