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
import torch
# from perplexity import calculate_perplexity
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig, BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, GPT2Config, AdamW
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoConfig
pd.options.display.max_colwidth = 1000

def precision(outputs, targets):
    return metrics.precision_score(targets, outputs, average = 'weighted')

def recall(outputs, targets):
    return metrics.recall_score(targets, outputs, average = 'weighted')

def processDataset(filepath, encoder_tokenizer, max_len, batch_size):
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

    # training_set = CustomDataset(train_dataset, encoder_tokenizer, decoder_tokenizer, max_len=max_len)
    # testing_set = CustomDataset(test_dataset, encoder_tokenizer, decoder_tokenizer, max_len=max_len)
    validation_set = CustomDataset(val_dataset, encoder_tokenizer, max_len=max_len)

    # train_params = {'batch_size': batch_size,
    #                 'shuffle': True,
    #                 'num_workers': 4
    #                 }

    # test_params = {'batch_size': batch_size,
    #                 'shuffle': True,
    #                 'num_workers': 4
    #                 }
    validation_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 4
                    }


    # training_loader = DataLoader(training_set, **train_params)
    # testing_loader = DataLoader(testing_set, **test_params)
    validation_loader = DataLoader(validation_set, **validation_params)
    return validation_loader
    
class CustomDataset(Dataset):
    
    def __init__(self, df, encoder_tokenizer, max_len):
        self.dataframe = df
        self.tokenizer = encoder_tokenizer
        self.description = df["dialogue"].apply(lambda x: x[0]).to_frame()
        self.department = df["doctor_faculty"]
        self.max_len = max_len

    def __len__(self):
        return len(self.description)

    def __getitem__(self, index):
        description = str(self.description.iloc[index])
        description = " ".join(description.split())

        inputs = self.tokenizer.encode_plus(
            description,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'department': torch.tensor(self.department.iloc[index], dtype=torch.long)
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
    # seqoutputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states = hidden_states, labels=decoder_input_ids) 
    # TODO: For actual validation, this is acceptable. But for testing, it would be cheating. We should have input_ids = "医生："
    # seqoutputs = self.decoder(input_embeds=hidden_states, attention_mask=decoder_attention_mask, encoder_hidden_states = hidden_states, labels=decoder_input_ids)
    
    # # logits has shape batch X seq_len X vocab_size
    # # logits = outputs.logits[:, -1, :]
 
    x = self.dropout(hidden_states[:,0,:]) # first element of last layer, batch_size * sequence length * embed_size (ORIGINAL)
    classificationLogits = self.classifier(x) #(8,10)
    # print(departmentlabels.shape)
    # departmentlabels_indices = departmentlabels.argmax(dim=1)
    # classificationAccuracy = metrics.accuracy_score(departmentlabels_indices.cpu().numpy(), classificationLogits.cpu().detach().argmax(dim=1).numpy())
    # classificationLoss = torch.nn.functional.cross_entropy(classificationLogits, departmentlabels.to(torch.float)) 

    return classificationLogits.cpu().detach().argmax(dim=1).numpy()     

def validation( model, testing_loader, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['department'].to(device, dtype = torch.float).argmax(dim=1)

            outputs = model(encoder_input_ids = ids, encoder_attention_mask = mask, departmentlabels = targets)
            outputPrediction = outputs
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputPrediction)

    return fin_outputs, fin_targets

device = 'cuda' if cuda.is_available() else 'cpu'
PATH_NAME = "./weights/BERT/"
def main():
    ## Sections of config
    MAX_LEN = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-05
    bert_checkpoint = "trueto/medbert-base-wwm-chinese"
    EPOCHS=5
    gpt_checkpoint = "uer/gpt2-chinese-cluecorpussmall"

    pretrained_model_parameters = "3_20_bert_med_4.bin"
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

    val_loader = processDataset(DatasetPath, encoder_tokenizer, MAX_LEN, BATCH_SIZE)

    ## load model
    print(' ----- loading data ------')
    model=CustomModel(checkpoint=bert_checkpoint, gpt_checkpoint=gpt_checkpoint, num_labels=10, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    if (pretrained_model_parameters != None):
        model.load_state_dict(torch.load(PATH_NAME + pretrained_model_parameters))
    
    print('---- start testing ----')
    model.eval()

    with torch.no_grad():
        for epoch in range(1):
            outputs, targets = validation(model, val_loader, 'cuda')
            print('outputs', outputs)
            print('targets', targets)
            print('outputs.shape', len(outputs))
            print('targets.shape', len(targets))
            outputs = np.array(outputs) >= 0.5
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
            precision_score = precision(outputs, targets)
            recall_score = recall(outputs, targets)
            
            print(f"Accuracy Score = {accuracy}")
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")
            print(f"Precision Score = {precision_score}")
            print(f"Recall Score = {recall_score}")
    

if __name__ == '__main__':
    main()