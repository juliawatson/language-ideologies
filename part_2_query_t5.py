from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
import time
import constants
import os
import tqdm
import csv
import datetime

def load_stimuli(data_path):
    """
    Load the stimuli from csv into a list of rows, each corresponding to a prompt
    """
    result = pd.read_csv(data_path, index_col="index")
    result["form_set"] = [eval(item) for item in result["form_set"]]
    return result


def set_up_model(imported=False):
    """
    Sets up variables to use model
    """
    global model
    global tokenizer
    global device
    global sentinel_0
    model = AutoModelForSeq2SeqLM.from_pretrained("google/" + constants.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained("google/" + constants.MODEL_NAME)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # specify gpu to use- check using nvidia-smi command on terminal
        torch.cuda.set_device(0)
    model = model.to(device)
    # device = "cpu"
    
    sentinel_0 = tokenizer("<extra_id_0>", return_tensors="pt").to(device).input_ids[0].tolist()[0]
    if imported:
        return model, tokenizer, device
    return None, None, None


def get_input_tokens(sentence, variant):
    """
    Convert input sentence. The general method uses a comparison of the full and masked sentences 
    to identify where the sentinel tokens should start. 
    I chose this method as opposed to searching for the variant tokens directly since there could be multiple 
    instances of the variant in the sentence without it being the target.
    """
        
    # Tokenize full and masked sentences and variant
    full_sentence = tokenizer(sentence.replace("[FORM]", variant), return_tensors="pt").to(device).input_ids[0].tolist()
    var_tokens = tokenizer(variant, return_tensors="pt").to(device).input_ids[0].tolist()[:-1]  # exclude last index, which is </s>]
    
    # Loop through to find the first different token
    for beginning_i in range(len(full_sentence) - len(var_tokens)):
        if full_sentence[beginning_i : beginning_i+len(var_tokens)] == var_tokens:
            break
    assert full_sentence[beginning_i] == var_tokens[0]     # check that the first different token is 
    
    # construct the first half of the final output
    final_input_ids = full_sentence[:beginning_i] + [sentinel_0] + full_sentence[beginning_i + len(var_tokens):]
    
    return torch.tensor([final_input_ids], dtype=torch.int).to(device)


def get_output_labels(input_tokens, variant):
    """
    Format the output label according to if the variant comes at the middle or end of the prompt.
    """
    # looking at second to last token
    if input_tokens.to("cpu")[0][-2].numpy() == sentinel_0:
        var_label = tokenizer("<extra_id_0>" + variant, return_tensors="pt").to(device).input_ids
        exclusion = {0, len(var_label[0]) - 1}     # the first token (sentinel) and </s> excluded in logprob
    else:
        var_label = tokenizer("<extra_id_0>" + variant + "<extra_id_1>", return_tensors="pt").to(device).input_ids
        exclusion = {0, len(var_label[0]) - 1, len(var_label[0]) - 2}   # exclude </s> and both sentinel tokens
        
    return var_label, exclusion


def compute_probabilities(logprobs, var_label, exclusion):
    """
    Compute the probabilities from the model
    """

    return sum(logprobs[token_index][var_label[0][token_index]].item() 
                for token_index in range(len(var_label[0])) if token_index not in exclusion) 


def logprob_variant(input_tokens, variant):
    """
    Query the model for the log probability of a given variant. 
    """
    var_label, exclusion = get_output_labels(input_tokens, variant)
    
    with torch.no_grad():
        output_logits = model(input_ids=input_tokens, labels=var_label).logits[0]
        
    logprobs = [torch.nn.functional.log_softmax(output_logits[i], dim=-1) for i in range(len(output_logits))]
    
    return compute_probabilities(logprobs, var_label, exclusion)  

    

def query_t5(output_path, loaded_sentences_df, model_name):
    """
    For each row in loaded_sentences_df, calculate the log probability of each variant 
    and save into output_path. 
    """
    fieldnames = ['name', 'item', 'form_set', 'perm', 'way_of_asking', 'context',
                  'prompt_text', 'model', 'logprobs_dict']
    
    # index,name,item,form_set,perm,way_of_asking,context,prompt_text
    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        start_time = time.time()
        
        for i,row in tqdm.tqdm(loaded_sentences_df.iterrows()):
            output_row = {}
            for fname in fieldnames:
                if fname in row:
                    output_row[fname] = row[fname]
            output_row["model"] = model_name
            
            form_dict = {}
            cur_input_tokens = None
            for form in row["form_set"]: 
                if cur_input_tokens is None: 
                    cur_input_tokens = get_input_tokens(row["prompt_text"], form)
                form_dict[form] = logprob_variant(cur_input_tokens, form)
            output_row["logprobs_dict"] = form_dict
            csv_writer.writerow(output_row)
        end_time = time.time()
        
        return end_time - start_time
            
        
def main(config):  
    """
    For each stimuli sentence, queries the model for probabilities of pronoun variants
    and saves the result in output_path file. This uses raw_path file as an intermediate.
    """
    print(f"Collecting data from model: {constants.MODEL_NAME}")
    print(f"config: {config}")
    date = str(datetime.datetime.now(datetime.timezone.utc))
    print(date)

    dirname = "/".join(config.split("/")[:-1])
    input_path = f"{dirname}/stimuli.csv"
    output_path = f"{dirname}/{constants.MODEL_NAME}/logprobs.csv"

    if not os.path.exists(f"{dirname}/{constants.MODEL_NAME}"):
        os.mkdir(f"{dirname}/{constants.MODEL_NAME}")

    input_sentences_df = load_stimuli(input_path)
    
    with open(f"{dirname}/{constants.MODEL_NAME}/running_metadata.txt", "a") as metadata:
        metadata.write(f"{date}\nTotal prompts: {str(len(input_sentences_df))}\nDevice: {device}\n")
        metadata.write("Total seconds: " + str(query_t5(output_path, input_sentences_df, model_name=constants.MODEL_NAME)))
      


if __name__ == "__main__":    
    assert("flan" in constants.MODEL_NAME)
    set_up_model()
    
    # main("analyses/experiment1/role-nouns-full/config.json")
    # main("analyses/experiment1/singular-pronouns-full/config.json")
    # main("analyses/experiment2/role-nouns-full/config.json")
    # main("analyses/experiment2/singular-pronouns-full/config.json")

    # main("analyses/experiment1/role-nouns-pilot/config.json")
    # main("analyses/experiment2/role-nouns-pilot/config.json")

    main("analyses/experiment1/role-nouns-expanded/config.json")
    main("analyses/experiment2/role-nouns-expanded/config.json")


    # ON VECTOR SERVER

    # experiment 1
    # role noun pilot (2000 prompts): 547.6 seconds
    #     * expanded set has 72000 prompts -> expect 19713.6 seconds

    # experiment 2
    # role noun pilot (360 prompts): 63.4 seconds
    #     * expanded set has 25600 prompts -> expect 4508.4 seconds

    # Total: 19713.6 + 4508.4 = 24222 seconds
    #     = 6.73 hours


    # ON CL SERVER

    # Testing- XL: 
    # --CPU--
    # 5: 40.97645926475525 (ram: 78 - 66)
    # 10: 80.34770703315735 (ram: 62 - 52)
    # Rate = 7.8 seconds per prompt
    # --GPU-- spock2
    # role noun pilot (2000): 387.7132270336151 (26 - 8)
    # singular pronoun pilot (3840): 800.1803021430969
    
    # experiment 2
    # role noun pilot (360): 66.50010561943054
    # singular pronoun pilot (576): 122.32357788085938
    
