"""Generates stimuli to feed into LLMs.

This is done for each domain: role nouns and singular pronouns.
The resulting stimuli are stored in stimuli_role_noun.csv and stimuli_singular_pronoun.csv.
The most important columns for querying LLMs are:
* prompt_text: Includes the prompt text, where [FORM] should be replaced with each of the relevant forms.
* form_set: The possible variants that can replace [FORM] in prompt_text.
"""

import collections
import csv
import itertools
import json
import pandas as pd

import constants
from common import load_config


def add_prompts(data, name, item, perm_form_set, completions_form_set, sentence_form,
                ways_of_asking, contexts):
    for way_of_asking, way_of_asking_template in ways_of_asking.items():
        for context, context_sentence in contexts.items():

            # If we list all the possible choices (as in the choices-all-terms condition),
            # then we need to consider all permutations.
            if context == "choices-all-terms":
                permutations = list(itertools.permutations(perm_form_set))
            else:
                permutations = ["N/A"]

            for perm in permutations:

                data["name"].append(name)
                data["item"].append(item)
                data["form_set"].append(completions_form_set)
                data["perm"].append(perm)

                data["way_of_asking"].append(way_of_asking)
                data["context"].append(context)

                # Replace [FORM] with ____ for metalinguistic prompts (all but direct)
                if way_of_asking == "direct":
                    curr_sentence_form = sentence_form
                else:
                    curr_sentence_form = sentence_form.replace("[FORM]", "____")

                # Insert sentence and name into the way of asking prompt
                if way_of_asking in ["likely_refer", "best_refer"]:
                    curr_prompt = way_of_asking_template.format(name, curr_sentence_form)
                else:
                    curr_prompt = way_of_asking_template.format(curr_sentence_form)

                # Insert name and permutation into context sentence where appropriate
                if context == "choices-all-terms":
                    curr_context_sentence = context_sentence.format(*perm)
                elif context in ["individual-declaration", "individual-declaration-prefer"]:
                    curr_context_sentence = context_sentence.format(name)
                else:
                    curr_context_sentence = context_sentence
                
                # Add the context sentence before the main prompt
                if context == "choices-all-terms" and way_of_asking != "direct":
                    curr_prompt = curr_prompt[0].lower() + curr_prompt[1:]
                    curr_prompt = f"{curr_context_sentence} Of these words, {curr_prompt}"
                elif context != "null_context":
                    curr_prompt = f"{curr_context_sentence} {curr_prompt}"

                data["prompt_text"].append(curr_prompt)


def generate_prompt_summary_sheet(name, item_label, data_path, ways_of_asking, contexts):
    df = pd.read_csv(data_path)
    df = df.loc[df["name"] == name]
    df = df.loc[df["item"] == item_label]
    
    output_path = data_path.replace(".csv", "_example.csv")

    with open(output_path, "w") as f:
        fieldnames = ["way_of_asking"] + list(contexts.keys())
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for way_of_asking in ways_of_asking:
            curr_df = df.loc[df["way_of_asking"] == way_of_asking]
            row = {
                row["context"]: row["prompt_text"]
                for _, row in curr_df.iterrows()
            }
            row["way_of_asking"] = way_of_asking
            csv_writer.writerow(row)


def main_singular_pronouns(config, output_dir):
    names = config["names"]
    forms = pd.read_csv(config["item_path"])

    data = collections.defaultdict(list)
    for name in names: 
        for _, row in forms.iterrows():
            sentence_form = row["sentence"].replace("[NOUN]", name)
            item_label = row["sentence"]
            add_prompts(
                data=data, 
                name=name, 
                item=item_label,
                perm_form_set=config["pronoun_sets_permutations"][row["form"]],
                completions_form_set=config["pronoun_sets_comprehensive"][row["form"]],
                sentence_form=sentence_form,
                ways_of_asking=config["ways_of_asking"],
                contexts=config["contexts"])

    df = pd.DataFrame(data)
    df.index.name = "index"
    output_path = f"{output_dir}/stimuli.csv"
    df.to_csv(output_path)

    generate_prompt_summary_sheet(
        name, item_label, output_path, config["ways_of_asking"], config["contexts"])


def main_role_nouns(config, output_dir):
    names = config["names"]
    noun_sets = config["role_nouns"]

    data = collections.defaultdict(list)
    for name in names: 
        for noun_set in noun_sets:
            determiner = "an" if tuple(noun_set) in constants.AN_NOUNS else "a"
            sentence_form = config["sentence_format"].format(name, determiner, "[FORM]")
            item_label = noun_set[0]
            
            add_prompts(
                data=data, 
                name=name, 
                item=item_label,
                perm_form_set=noun_set,
                completions_form_set=noun_set,
                sentence_form=sentence_form,
                ways_of_asking=config["ways_of_asking"],
                contexts=config["contexts"])

    df = pd.DataFrame(data)
    df.index.name = "index"
    output_path = f"{output_dir}/stimuli.csv"
    df.to_csv(output_path)

    generate_prompt_summary_sheet(
        name, item_label, output_path, config["ways_of_asking"], config["contexts"])
    

def main_singular_pronouns_camilliere_stimuli_exact_rerun(config, output_dir):
    forms = pd.read_csv(config["item_path"])

    data = collections.defaultdict(list)
    for _, row in forms.iterrows():
        sentence_form = row["sentence"].replace(row["form"], "[FORM]")
        assert "[FORM]" in sentence_form
        item_label = row["itm"]

        # We focus on singular cases, since plural (and some quantified) cases
        # have verb agreement that only works with "they" (not compatible with
        # gendered singular pronouns like he/she)
        if row["cond"] not in ["plural", "quantifier"]:
            add_prompts(
                data=data,
                name=row["antecedent"],
                item=item_label,
                perm_form_set=config["pronoun_sets_permutations"][row["form"]],
                completions_form_set=config["pronoun_sets_comprehensive"][row["form"]],
                sentence_form=sentence_form,
                ways_of_asking=config["ways_of_asking"],
                contexts=config["contexts"])

    df = pd.DataFrame(data)
    df.index.name = "index"
    output_path = f"{output_dir}/stimuli.csv"
    df.to_csv(output_path)


def main(config_path):
    # Load the config
    config = load_config(config_path)
    config_dir = "/".join(config_path.split("/")[:-1])
    
    # Run the correct main function for the domain
    if config["domain"] == "role_nouns":
        main_role_nouns(config, config_dir)
    elif config["domain"] == "singular_pronouns":
        main_singular_pronouns(config, config_dir)
    elif config["domain"] == "singular_pronouns_camilliere_stimuli_exact_rerun":
        main_singular_pronouns_camilliere_stimuli_exact_rerun(config, config_dir)
    else:
        raise ValueError(f"Domain type not supported: {config['domain']}")


if __name__ == "__main__":
    # main("analyses/experiment1/role-nouns-full/config.json")
    # main("analyses/experiment1/singular-pronouns-full/config.json")

    # main("analyses/experiment1/role-nouns-full-extra-conditions/config.json")
    # main("analyses/experiment1/singular-pronouns-full-extra-conditions/config.json")
    # main("analyses/experiment1/singular-pronouns-full-camilliere-stimuli-exact-rerun/config.json")

    # main("analyses/experiment1/singular-pronouns-full-null-context-only-to-be-used-to-correction/config.json")

    # main("analyses/experiment2/role-nouns-pilot-additional-prompts-2/config.json")
    # main("analyses/experiment2/singular-pronouns-pilot-additional-prompts-2/config.json")

    # main("analyses/experiment2/role-nouns-full/config.json")
    # main("analyses/experiment2/singular-pronouns-full/config.json")

    main("analyses/experiment1/role-nouns-expanded/config.json")
    main("analyses/experiment2/role-nouns-expanded/config.json")
