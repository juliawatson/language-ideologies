from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import time
import constants
import os
import tqdm
import csv
import datetime
import numpy as np


# To download models:
# cd /scratch/ssd004/scratch/jwatson
# huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir meta-llama/Meta-Llama-3-8B
# huggingface-cli download meta-llama/Meta-Llama-3.1-8B --local-dir meta-llama/Meta-Llama-3.1-8B
# huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir meta-llama/Llama-2-7b-hf


MODEL_NAME_TO_MODEL_PATH = {
    "llama-3-8B": "/scratch/ssd004/scratch/jwatson/meta-llama/Meta-Llama-3-8B",
    "llama-3.1-8B": "/scratch/ssd004/scratch/jwatson/meta-llama/Meta-Llama-3.1-8B",
    "llama-2-7B":  "/scratch/ssd004/scratch/jwatson/meta-llama/Llama-2-7b-hf"
}

MODEL_NAME_TO_WHITESPACE_CHARACTER = {
    "llama-3-8B": "Ġ",
    "llama-3.1-8B": "Ġ",
    "llama-2-7B": "▁"
}
WHITESPACE_CHARACTER = MODEL_NAME_TO_WHITESPACE_CHARACTER[constants.MODEL_NAME]


def load_stimuli(data_path):
    """
    Load the stimuli from csv into a list of rows, each corresponding to a prompt
    """
    result = pd.read_csv(data_path, index_col="index")
    result["form_set"] = [eval(item) for item in result["form_set"]]
    return result


def set_up_model(imported=False, device_name="cuda"):
    """
    Sets up variables to use model
    """
    global model
    global tokenizer
    global device
    device = device_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_MODEL_PATH[constants.MODEL_NAME])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_TO_MODEL_PATH[constants.MODEL_NAME],
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
        
    if imported:
        return model, tokenizer, device
    return None, None, None


def compute_probability(sentence_logprobs, string_tokens, variant, way_of_asking):

    def convert_tokens_to_string(token_list):
        return "".join(token_list).replace(WHITESPACE_CHARACTER, " ").strip()

    if way_of_asking == "direct":  # use all tokens in the sentence
        token_logprobs = sentence_logprobs
    else:
        # Figure out how many tokens make up the variant/continuation
        i = len(string_tokens) - 1
        variant_str = convert_tokens_to_string(string_tokens[i])
        while variant_str != variant and i > 0:
            i -=1
            variant_str = convert_tokens_to_string(string_tokens[i:])
        assert variant_str == variant, f"{variant_str} != {variant}; string_tokens={string_tokens}"
        token_logprobs = sentence_logprobs[i:]
    
    return np.sum(token_logprobs)


def get_sentence_token_logprobs(logprobs, sentence_tokens):
    """For each token in sentence_tokens, extract that
    token's probability in logprobs.
    
    input shape: [n_tokens, vocab_size]
    output shape: [n_tokens]
    """
    result = []
    # Note that:
    # - logprobs contains probability distribution over tokens *after*
    #   the token at that position in sentence_tokens.input_ids.
    # - sentence_tokens starts with <|start_of_text|>.
    # This means the probabability of the first word in the sentence (after
    # <|start_of_text|>) will be at logprobs[0, sentence_tokens.input_ids[0][1]]
    # This is why we iterate over sentence_tokens.input_ids[0], starting at
    # index 1.
    for token_i, vocab_i in enumerate(sentence_tokens["input_ids"][0][1:]):
        result.append(logprobs[token_i, vocab_i].item())
    return np.array(result)


def run_model(prompt_text, variant, way_of_asking):
    """
    Query the model for the log probability of a given variant. 
    """
    sentence = prompt_text.replace("[FORM]", variant)

    # Tokenize the sentence
    sentence_tokens = tokenizer(sentence, return_tensors="pt")
    sentence_tokens.to("cuda")

    # Feed the sentence into the model
    with torch.no_grad():
        # shape: [n_tokens, vocab_size]
        output_logits = model(input_ids=sentence_tokens.input_ids).logits[0]

    # normalize to get probabilities -- make this a testable function
    logprobs = torch.log_softmax(output_logits, axis=1)
    # assert torch.allclose(torch.sum(torch.exp(logprobs), axis=1), torch.tensor(1.).to("cuda"))
    
    # use normalized probabilities to compute the probability of the variant -- make this a testable function
    sentence_logprobs = get_sentence_token_logprobs(logprobs, sentence_tokens)
    string_tokens = tokenizer.convert_ids_to_tokens(sentence_tokens["input_ids"][0])[1:]
    return compute_probability(sentence_logprobs, string_tokens, variant, way_of_asking)  


def query_llama(output_path, loaded_sentences_df, model_name):
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

            # print(row["prompt_text"])
            # print(row["form_set"])
            # Assume you want to use language that is correct. The best word to complete the sentence "Finley is a ____." is [FORM]
            # ['flight attendant', 'steward', 'stewardess']
            
            form_dict = {}
            cur_input_tokens = None
            for form in row["form_set"]:
                form_dict[form] = run_model(row["prompt_text"], form, row["way_of_asking"])
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
        metadata.write("Total seconds: " + str(query_llama(output_path, input_sentences_df, model_name=constants.MODEL_NAME)))
      


if __name__ == "__main__":    
    assert("llama" in constants.MODEL_NAME.lower())
    set_up_model()
    
    # main("analyses/experiment1/role-nouns-pilot/config.json")
    # main("analyses/experiment2/role-nouns-pilot/config.json")

    # main("analyses/experiment1/role-nouns-full/config.json")
    # main("analyses/experiment1/singular-pronouns-full/config.json")
    # main("analyses/experiment2/role-nouns-full/config.json")
    # main("analyses/experiment2/singular-pronouns-full/config.json")

    main("analyses/experiment1/role-nouns-expanded/config.json")
    main("analyses/experiment2/role-nouns-expanded/config.json")

    # Estimating time with GPU on vector server for llama-3-8B
    
    # experiment 1
    # role noun pilot (2000 prompts): 297 seconds
    #     * full experiment has 25200 prompts -> expect 3742.2 seconds
    #     * expanded set has 72000 prompts -> expect 10692 seconds
    # singular pronoun pilot (3840 prompts): 625 seconds
    #     * full experiment has 32000 prompts -> expect 5208.3 seconds

    # experiment 2
    # role noun pilot (360 prompts): 54 seconds
    #     * full experiment has 8960 prompts -> expect 1344 seconds
    #     * expanded set has 25600 prompts -> expect 3840
    # singular pronoun pilot (576 prompts): 93 seconds
    #     * full experiment has 25600 prompts -> expect 4133.3 seconds

    # Total time for original "full" experiments: 3742.2 + 5208.3 + 1344 + 4133.3 = 14427.8 seconds
    #    = 4.008 hours

    # Total time for role nouns expanded set:  10692 + 3840 = 14532 seconds
    #    = 4.037 hours

