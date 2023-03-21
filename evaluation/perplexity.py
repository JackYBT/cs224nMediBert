import torch
import tqdm
import util
import numpy as np
from CS224N_3_14_BERT_GPT_seq2seq import CustomModel, processDataset
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoConfig
import evaluation.perplexity as perplexity
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig, BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, GPT2Config, AdamW
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoConfig

def check_compatibility(state_dict, model):
    model_dict = model.state_dict()

    for key in state_dict.keys():
        if key not in model_dict:
            print(f"Key '{key}' not found in the model.")
            return False

        if state_dict[key].shape != model_dict[key].shape:
            print(f"Shape mismatch for '{key}': expected {model_dict[key].shape}, but got {state_dict[key].shape}")
            return False

    return True

def filter_extra_keys(state_dict, model):
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    return filtered_state_dict

def calculate_perplexity(
    batch_size=1,
    decoder_path='decoder.pth',
    model = None,
    test_dataloader=None,
    device='cuda'):
    # make sure your model is on GPU
    device = torch.device(device)

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    model = model
    state_dict = torch.load(decoder_path)

    state_dict = filter_extra_keys(state_dict, model)

    if check_compatibility(state_dict, model):
        model.load_state_dict(state_dict)
    else:
        print("The model is not compatible with the checkpoint.")
        print("model_statedict_keys()", model.state_dict().keys())
        print("state_dict.keys()", state_dict.keys())
        return
    print(f'load from {decoder_path}')
    model = model.to(device)
    model.eval()
    print('load success')
    #------------------------END LOAD MODEL--------------

    test_dataloader = test_dataloader
    #------------------------END LOAD VAL DATA--------------

    perplexity = 0
    batch_count = 0
    print('start calculate the test perplexity....')
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for _, data in enumerate(test_dataloader, 0):
            encoder_input = data['input_ids'].to(device, dtype = torch.long)
            mask_encoder_input = data['input_mask'].to(device, dtype = torch.long)
            decoder_input = data['output_ids'].to(device, dtype = torch.long)
            mask_decoder_input = data['output_mask'].to(device, dtype=torch.long)

            outputs = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

            logits = outputs.logits
            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            num_tokens = target_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            batch_count += 1
            
            if batch_count % 100 == 0:
                print(f'batch: {_}, currentPerplexity for this batch: {np.exp(loss.item())}')
                print(f'batch: {_}, perplexity so far: {np.exp(total_loss / total_tokens)}')



    # Calculate the perplexity for the entire dataset
    perplexity = np.exp(total_loss / total_tokens)
    print(f'Perplexity: {perplexity}')

if __name__ == '__main__':
    BATCH_SIZE = 8
    PATH_NAME = "./"
    MAX_LEN = 200
    bert_checkpoint = "trueto/medbert-base-wwm-chinese"

    tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
    tokenizer.model_max_len=512
    gpt_checkpoint = "uer/gpt2-chinese-cluecorpussmall"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=CustomModel(checkpoint=bert_checkpoint, gpt_checkpoint=gpt_checkpoint, num_labels=10).to(device)
    training_loader, testing_loader = processDataset('Dataset/validate_data.json', tokenizer, MAX_LEN, BATCH_SIZE)
    calculate_perplexity(batch_size=BATCH_SIZE, decoder_path="{PATH_NAME}/bert_gpt_validate_data_testingloader_seq2seq.bin".format(PATH_NAME=PATH_NAME), model=model, test_dataloader=testing_loader)