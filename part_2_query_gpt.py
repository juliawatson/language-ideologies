# Requires openai==0.28

import pandas as pd
import tqdm
import csv
import numpy as np
import datetime
import os
import openai
import time

import constants


def load_stimuli(data_path):
    """
    Load the stimuli from csv into a list of rows, each corresponding to a prompt
    """
    result = pd.read_csv(data_path, index_col="index")
    result["form_set"] = [eval(item) for item in result["form_set"]]
    return result


def set_up_api():
    """
    Sets up connection to the API, need to put OPEN_API_KEY as a variable in the terminal before running
    """
    # export OPENAI_API_KEY="INSERT_KEY_HERE"
    # Set up access to the api
    openai.organization = "org-tl7YMxuVpm9XVcU5fHxH2QDh"
    openai.api_key = os.getenv("OPENAI_API_KEY")    

                
def query_gpt(raw_path, loaded_sentences_df, model):
    """
    This method is only compatible with gpt model. 
    For each stimuli sentence, queries gpt-3's api for probabilities of "they" forms amongst other information. 
    The raw output of api queries is saved in raw_path file.  
    """
    set_up_api()

    with open(raw_path, "w") as f:
        fieldnames = list(loaded_sentences_df.columns) + [
            "variant", "finish_reason", "logprobs", "text", "created", "id", 
            "model", "object", "usage", "timestamp", "full_output"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        for i, row in tqdm.tqdm(loaded_sentences_df.iterrows()): 
            base_sentence = row["prompt_text"]
            for form in row["form_set"]:
                sentence = base_sentence.replace("[FORM]", form)
                # start_time = time.time()
                completed = False
                while not completed: 
                    try:
                        output = openai.Completion.create(
                            model=model,
                            prompt=sentence,
                            logprobs=0,
                            max_tokens=0,                 # This says not to add any more tokens at the end
                            echo=True)                    # This says to give us the probabilities of the words in the prompt
                        completed = True
                    except openai.error.RateLimitError as e: 
                        print(f"Index:{i}\n", e, "\n")
                        time.sleep(10)
                        completed = False
                    except Exception as e: 
                        time.sleep(10)
                        print(f"Index:{i}\n", e, "\n")
                        completed = False
                row = dict(row)
                row["variant"] = form
                row["finish_reason"] = output["choices"][0]["finish_reason"]
                row["logprobs"] = dict({
                    "text_offset": list(output["choices"][0]["logprobs"]["text_offset"]), 
                    "token_logprobs": list(output["choices"][0]["logprobs"]["token_logprobs"]), 
                    "tokens": list(output["choices"][0]["logprobs"]["tokens"]),
                })
                row["text"] = output["choices"][0]["text"]
                row["created"] = output["created"]
                row["id"] = output["id"]
                row["model"] = output["model"]
                row["object"] = output["object"]
                row["usage"] = dict({
                    "prompt_tokens": output["usage"]["prompt_tokens"], 
                    "total_tokens": output["usage"]["total_tokens"]})
                row["full_output"] = " ".join(str(dict(output)).replace("\n", " ").split())
                row["timestamp"] = str(datetime.datetime.now(datetime.timezone.utc))
                csv_writer.writerow(row)

                # No longer relevant, since our rate limit is much higher.
                # Sleep to ensure that 60 queries a minute is not breached for rate limit-
                # Source: https://platform.openai.com/docs/guides/rate-limits/overview
                # Can instead sleep for one second, I chose to calculate the remaining time to save runtime
                # end_time = time.time()
                # remaining = 1 - (end_time - start_time)
                # if remaining > 0:
                #     time.sleep(remaining)


def reformat_raw_gpt(raw_path, output_path):
    """
    This method is only compatible with gpt-3. 
    This method reformats the results of api query stored in raw_path file to the same format as all other models. 
    """
    raw_data = pd.read_csv(raw_path)
    raw_data["form_set"] = [eval(item) for item in raw_data["form_set"]]

    fieldnames = [
        'name', 'item', 'form_set', 'perm', 'way_of_asking', 'context',
        'prompt_text', 'model', 'logprobs_dict']
    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for _, item_df in raw_data.groupby(["prompt_text", "item"]):

            # Check that all the relevant forms are present
            assert set(item_df["variant"]) == set(list(item_df["form_set"])[0])

            variant_logprobs = {}
            for _, variant_row in item_df.iterrows():
                variant = variant_row["variant"]
                logprobs_dict = eval(variant_row["logprobs"])
                if variant_row["way_of_asking"] == "direct":
                    token_logprobs = logprobs_dict["token_logprobs"][1:]  # We skip index 0 bc it's always None
                else:
                    # Figure out how many tokens make up the variant/continuation
                    i = len(logprobs_dict["tokens"]) - 1
                    variant_str = logprobs_dict["tokens"][i]
                    while variant_str.strip() != variant:
                        i -=1                        
                        variant_str = logprobs_dict["tokens"][i] + variant_str
                    token_logprobs = logprobs_dict["token_logprobs"][i:]

                variant_logprobs[variant] = np.sum([token for token in token_logprobs])

            # Add all relevant fields from fieldnames to row
            output_row = {}
            for fname in fieldnames:
                if fname in variant_row:
                    output_row[fname] = variant_row[fname]

            output_row["logprobs_dict"] = variant_logprobs
            csv_writer.writerow(output_row)

    
def main(config):  
    """
    For each stimuli sentence, queries the model for probabilities of pronoun variants
    and saves the result in output_path file. This uses raw_path file as an intermediate.
    """
    print(f"Collecting data from model: {constants.MODEL_NAME}")
    print(f"config: {config}")

    dirname = "/".join(config.split("/")[:-1])
    input_path = f"{dirname}/stimuli.csv"
    raw_path = f"{dirname}/{constants.MODEL_NAME}/raw.csv"
    output_path = f"{dirname}/{constants.MODEL_NAME}/logprobs.csv"

    if not os.path.exists(f"{dirname}/{constants.MODEL_NAME}"):
        os.mkdir(f"{dirname}/{constants.MODEL_NAME}")

    # input_sentences_df = load_stimuli(input_path)
    # query_gpt(raw_path, input_sentences_df, model=constants.MODEL_NAME)
    
    # Filter output to format used for rest of the models
    reformat_raw_gpt(raw_path, output_path)

    
if __name__ == "__main__":
    main("analyses/experiment1/role-nouns-full/config.json")
    main("analyses/experiment1/singular-pronouns-full/config.json")

    main("analyses/experiment2/role-nouns-full/config.json")
    main("analyses/experiment2/singular-pronouns-full/config.json")

