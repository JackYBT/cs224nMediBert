from torch import cuda, nn
import time
import json
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from evaluation.bleu.bleu import Bleu
from evaluation.rouge import Rouge
import wandb
wandb.init(project="CS224N_3_16_BERT_GPT_multitask")
import torch
# from perplexity import calculate_perplexity
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig, BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, GPT2Config, AdamW
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoConfig, pipeline
pd.options.display.max_colwidth = 1000

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def processDataset(filepath, encoder_tokenizer, decoder_tokenizer, max_len, batch_size):
        ## Process data
    f = open(filepath)
    data = json.load(f)
    f.close()
    df = pd.DataFrame(data)

    #map each department to an index 
    departments = df['doctor_faculty'].unique()
    d2ind = {departments[i]:i for i in range(len(departments))}
    ind2d = {i:departments[i] for i in range(len(departments))}

    for i, row in df.iterrows():
        one_hot = np.zeros(len(departments))
        one_hot[d2ind[row['doctor_faculty']]]=1
        df.at[i, 'doctor_faculty'] = one_hot

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

    training_set = CustomDataset(train_dataset, encoder_tokenizer, decoder_tokenizer, max_len=max_len)
    testing_set = CustomDataset(test_dataset, encoder_tokenizer, decoder_tokenizer, max_len=max_len)
    validation_set = CustomDataset(val_dataset, encoder_tokenizer, decoder_tokenizer, max_len=max_len)

    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 4
                    }

    test_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 4
                    }
    validation_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 4
                    }


    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    validation_loader = DataLoader(validation_set, **validation_params)
    return training_loader, testing_loader, validation_loader
    
class CustomDataset(Dataset):

    
    def __init__(self, df, encoder_tokenizer, decoder_tokenizer, max_len):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.dataframe = df
        self.patient = df["dialogue"].apply(lambda x: x[0]).to_frame()
        self.doc = df["dialogue"].apply(lambda x: x[1]).to_frame()
        self.department = df["doctor_faculty"]
        self.max_len = max_len

    def __len__(self):
        return len(self.patient)

    def __getitem__(self, index):
        # input = str(self.patient[index])
        input = self.patient.iloc[index]["dialogue"]
 
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

        output = self.doc.iloc[index]["dialogue"]
        outputs = self.decoder_tokenizer.encode_plus(
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

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'input_mask': torch.tensor(input_mask, dtype=torch.long),
            'input_token_type_ids': torch.tensor(input_token_type_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'output_mask': torch.tensor(output_mask, dtype=torch.long),
            'output_token_type_ids': torch.tensor(output_token_type_ids, dtype=torch.long),
            'department': torch.tensor(self.department[index], dtype=torch.long)
        }



class CustomModel(torch.nn.Module):
  def __init__(self,checkpoint, gpt_checkpoint, encoder_tokenizer, decoder_tokenizer, num_labels,temperature=0.5, dropout_rate = 0.1): 
    super(CustomModel,self).__init__() 
    self.num_labels = num_labels 
    self.temperature = temperature
    self.dropout_rate = dropout_rate
    self.encoder_tokenizer = encoder_tokenizer
    self.decoder_tokenizer = decoder_tokenizer

    #Load Model with given checkpoint and extract its body
    myConfig = AutoConfig.from_pretrained(checkpoint,output_hidden_states=True)
    myConfig.problem_type = "multi_label_classification"
    # myConfig.temperature = self.temperature
    myConfig.output_attentions = True

    self.encoder = AutoModel.from_pretrained(checkpoint,config=myConfig)
    self.decoder = GPT2LMHeadModel.from_pretrained(gpt_checkpoint, add_cross_attention=True).to(device)  

    self.dropout = torch.nn.Dropout(self.dropout_rate) 
    self.classifier = torch.nn.Linear(self.encoder.config.hidden_size,num_labels) # load and initialize weights
    self.criterion = torch.nn.CrossEntropyLoss() # define loss function

  def forward(self, encoder_input_ids=None, encoder_attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,departmentlabels=None):
    #Extract outputs from the body

    outputs = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, output_hidden_states=True)
    # select the 12th layer 
    hidden_states = outputs.hidden_states[-1]
    
    # print(' ----- in forward() -----')

    # This feeds the last hidden state of the encoder into the decoder    

    # I SUSPECT THAT INPUT_IDS NEED TO BE UPDATED TO "医生：and be constantly updated as we add to it?"
    seqoutputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states = hidden_states, labels=decoder_input_ids) 
    # TODO: For actual validation, this is acceptable. But for testing, it would be cheating. We should have input_ids = "医生："
    # seqoutputs = self.decoder(input_embeds=hidden_states, attention_mask=decoder_attention_mask, encoder_hidden_states = hidden_states, labels=decoder_input_ids)
    
    # # logits has shape batch X seq_len X vocab_size
    # # logits = outputs.logits[:, -1, :]
 
    x = self.dropout(hidden_states[:,0,:]) # first element of last layer, batch_size * sequence length * embed_size (ORIGINAL)
    classificationLogits = self.classifier(x) #(8,10)
    departmentlabels_indices = departmentlabels.argmax(dim=1)
    classificationAccuracy = metrics.accuracy_score(departmentlabels_indices.cpu().numpy(), classificationLogits.cpu().detach().argmax(dim=1).numpy())
    f1_score_micro = metrics.f1_score(departmentlabels_indices.cpu().numpy(), classificationLogits.cpu().detach().argmax(dim=1).numpy(), average='micro')
    f1_score_macro = metrics.f1_score(departmentlabels_indices.cpu().numpy(), classificationLogits.cpu().detach().argmax(dim=1).numpy(), average='macro')
    precision_score = metrics.precision_score(departmentlabels_indices.cpu().numpy(), classificationLogits.cpu().detach().argmax(dim=1).numpy(), average='macro')
    recall_score = metrics.recall_score(departmentlabels_indices.cpu().numpy(), classificationLogits.cpu().detach().argmax(dim=1).numpy(), average='macro')
    
    #create a onehot vector for the labels
    departmentlabels_onehot = torch.nn.functional.one_hot(departmentlabels_indices, num_classes=self.num_labels)
    
    # roc_auc = metrics.roc_auc_score(departmentlabels_indices, classificationLogits.cpu().detach().argmax(dim=1).numpy(), average='micro')
    classificationLoss = torch.nn.functional.cross_entropy(classificationLogits, departmentlabels.to(torch.float)) 


    return hidden_states, seqoutputs, classificationLoss, classificationAccuracy, f1_score_micro, f1_score_macro, precision_score, recall_score


def generate_gts(curBatchIndex, batch_data, tokenizer):
    # takes in a pandas df with patient's question and doctor's answer 
    # returns a dictionary of index -> doctor's groundtruth answer 
    gts = {}
    encoder_input = batch_data['output_ids'].to(device, dtype = torch.long)
    answers = tokenizer.batch_decode(encoder_input, skip_special_tokens=True, clean_up_tokenization_space=True)
    
    for i, answer in enumerate(answers):
        curIndex = curBatchIndex*8 + i
        gts = {**{curIndex:[answer]}, **gts}
    return gts

def eval_prep(model, validation_loader):
    gts = {}
    res = {}
    for _, batch_data in enumerate(validation_loader, 0):
        # print('---- generating gts ------')
        gts = {**gts, **generate_gts(_, batch_data, model.decoder_tokenizer)}
        # print(' ---- generating res -----')

        encoder_input = batch_data['input_ids'].to(device, dtype = torch.long)
        mask_encoder_input = batch_data['input_mask'].to(device, dtype = torch.long)
        decoder_input = batch_data['output_ids'].to(device, dtype = torch.long)
        mask_decoder_input = batch_data['output_mask'].to(device, dtype=torch.long)
        departmentlabels = batch_data['department'].to(device, dtype=torch.long)

        enc_hs, outputs, classificationLoss, classificationAccuracy= model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, departmentlabels)

        sample_output = torch.argmax(outputs.logits, dim = -1).squeeze()

        for index, batch_chinese_text in enumerate(sample_output):
            chinese_text = model.decoder_tokenizer.decode(batch_chinese_text, skip_special_tokens=True, clean_up_tokenization_space=True)
            curIndex = 8*_ + index
            res[curIndex] = [chinese_text]

            assert(len(res[curIndex]) == 1)
        # print("gts.keys()", gts.keys())
        # print("res.keys()", res.keys())

        # decoder_prompt = torch.tensor([model.decoder_tokenizer.encode("医生：", 
        #         add_special_tokens=True,
        #         max_length=200,
        #         pad_to_max_length=True)] * len(batch_data['output_ids'])).to(device, dtype=torch.long)
        # res = generate_res(model, batch_data, model.decoder_tokenizer, decoder_prompt, hidden_states, batch_data['output_mask'].to(device, dtype=torch.long))
    return gts, res


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

def train(optimizer, model, training_loader, validation_loader, device, num_epochs, learning_rate = 0.1):
#------------------------START TRAINING-------------------
    update_count = 0

    start = time.time()
    print('start training....')
    for epoch in range(num_epochs):
        #------------------------training------------------------
        model.train()
        epoch_classification_loss = 0
        epoch_seq2seq_loss = 0
        epoch_classification_accuracy = 0

        total_losses = 0
        times = 0

        averageClassificationAccuracy = 0
        averageF1ScoreMicro = 0
        averageF1ScoreMacro = 0
        averagePrecision = 0
        averageRecall = 0

        for update_count, data in enumerate(training_loader):
            encoder_input = data['input_ids'].to(device, dtype = torch.long)
            mask_encoder_input = data['input_mask'].to(device, dtype = torch.long)
            #encoder_token_type_ids = data['input_token_type_ids'].to(device, dtype = torch.long)
            decoder_input = data['output_ids'].to(device, dtype = torch.long)
            mask_decoder_input = data['output_mask'].to(device, dtype=torch.long)
            departmentlabels = data['department'].to(device, dtype=torch.long)
            #decoder_token_type_ids = data['output_token_type_ids'].to(device, dtype=torch.long)
            model.eval()
            with torch.no_grad():
                enc_hs, outputs, classificationLoss, classificationAccuracy, f1_score_micro, f1_score_macro, precision, recall = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, departmentlabels)

            averageClassificationAccuracy += classificationAccuracy
            averageF1ScoreMicro += f1_score_micro
            averageF1ScoreMacro += f1_score_macro
            averagePrecision += precision
            averageRecall += recall
            print(f"Accuracy Score = {classificationAccuracy}")
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")
            print(f"Precision Score = {precision}")
            print(f"Recall Score = {recall}")


            wandb.log({"Accuracy": averageClassificationAccuracy/(update_count+1), "F1 Score (Micro)": averageF1ScoreMicro/(update_count+1), "F1 Score (Macro)": averageF1ScoreMacro/(update_count+1), "Precision": averagePrecision/(update_count+1), "Recall": averageRecall/(update_count+1)})
     
            seq_loss = outputs.loss
            curLoss = classificationLoss + seq_loss
            # curLoss.backward()

            epoch_classification_loss += classificationLoss.item()
            epoch_seq2seq_loss += seq_loss.item()
            epoch_classification_accuracy += classificationAccuracy.item()

            total_losses += curLoss.item()
            times += 1

            optimizer.step()
            optimizer.zero_grad()
            # print("update_count: ", update_count)
            if update_count % 250 == 0:
                print('-'*20 + f'epoch {epoch}' + '-'*20)
                print(f'update count: {update_count}')
                print(f'total loss: {total_losses / times}')
                print(f'seq loss: {seq_loss}')
                print(f'classification loss: {classificationLoss}')
                print(f'total current Loss: {curLoss}')
                print(f'classification accuracy: {classificationAccuracy}')
                print(f'classification accuracy so far: {epoch_classification_accuracy / (update_count + 1)}')
                print('-'*20)
                times = 0
                total_losses = 0
                # wandb.log({'current classificationLoss': classificationLoss, 'current seq2seqLoss': seq_loss, 'current totalLoss': curLoss, 'current classificationAccuracy': classificationAccuracy})
                # wandb.log({'cumulative classificationLoss': epoch_classification_loss / (update_count + 1), 'cumulative seq2seqLoss': epoch_seq2seq_loss / (update_count + 1), 'cumulative totalLoss': (epoch_classification_loss + epoch_seq2seq_loss) / (update_count + 1), 'cumulative classificationAccuracy': epoch_classification_accuracy / (update_count + 1)})
            # if update_count % 45000 == 0:
            #     gts, res = eval_prep(model, encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, outputs, departmentlabels, update_count)
            #     sorted_keys_gts = sorted(gts.keys(), reverse=True)

            #     for index, key in enumerate(sorted_keys_gts):
            #         print(f"Input patient symptoms {index}:", model.encoder_tokenizer.decode(encoder_input[index], skip_special_tokens=True, clean_up_tokenization_space=True))
            #         print(f"gts {key}", gts[key])
            #         print(f"res {key}", res[key]) 
            # if update_count % 50000 == 0:
            #     print('------ evaluating ------')
            #     model.eval()
            #     # passing in the testing set, get perplexity values on training
            #     # the below function takes in gts and res, both dict()
            #     # they should have the same keys from id -> text
            #     with torch.no_grad():
            #         validation(model, validation_loader, device)
                    
            #     model.train()

        end = time.time()
        print('-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'time: {(end - start)}')
        print(f'total loss: {total_losses / times}')
        start = end

        torch.save(model.state_dict(), "{PATH_NAME}/{destinationFileName}_{EPOCH}.bin".format(PATH_NAME=PATH_NAME,destinationFileName=destinationFileName, EPOCH=epoch))
    
    # wandb.finish()

PATH_NAME = "./weights/Multitask"
device = 'cuda' if cuda.is_available() else 'cpu'
destinationFileName = '3_20_bert_gpt_multitask_allpatient_trained_from3_18GPT'

def main():
    ## Sections of config
    MAX_LEN = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-05
    bert_checkpoint = "trueto/medbert-base-wwm-chinese"
    EPOCHS=10
    gpt_checkpoint = "uer/gpt2-chinese-cluecorpussmall"

    pretrained_model_parameters = "/3_18_bert_gpt_multitask_allpatient_1.bin"
    DatasetPath = 'Dataset/patient_data.json'
    # pretrained_gpt_parameters = "./weights/GPT/3_19_GPT_gen_3.bin"

    encoder_tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint, max_len=512)
    decoder_tokenizer = AutoTokenizer.from_pretrained(gpt_checkpoint, max_len=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # local_saved_model = PATH_NAME + "3_17_bert_gpt_multitask_allpatient_1"
    # local_model = CustomModel(checkpoint=bert_checkpoint, gpt_checkpoint=gpt_checkpoint, num_labels=10, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer).to(device)
    # local_model.load_state_dict(torch.load(local_saved_model))

    # text_generator = TextGenerationPipeline(local_model, decoder_tokenizer)
    # text_generated = text_generator()

    training_loader, testing_loader, validation_loader = processDataset(DatasetPath, encoder_tokenizer, decoder_tokenizer, MAX_LEN, BATCH_SIZE)

    ## load model
    print(' ----- loading data ------')
    model=CustomModel(checkpoint=bert_checkpoint, gpt_checkpoint=gpt_checkpoint, num_labels=10, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    if (pretrained_model_parameters != None):
        model.load_state_dict(torch.load(PATH_NAME + pretrained_model_parameters))
    # elif (pretrained_gpt_parameters != None):
    #     model.decoder.load_state_dict(torch.load(pretrained_gpt_parameters), strict=False)
    # print('---- start testing ----')

    # validation(model, testing_loader, decoder_tokenizer, device)
    
    ## start training
    print('---- starting training----')
    train(optimizer, model, testing_loader, validation_loader, device, EPOCHS, LEARNING_RATE)

    # save the model after training
    print(' ----- saving model -----')
    torch.save(model.state_dict(), "{PATH_NAME}/{destinationFileName}.bin".format(PATH_NAME=PATH_NAME, destinationFileName=destinationFileName))
    print("model is saved to {PATH_NAME}/{destinationFileName}.bin".format(PATH_NAME=PATH_NAME, destinationFileName=destinationFileName))

if __name__ == '__main__':
    main()