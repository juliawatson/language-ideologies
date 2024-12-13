import collections
import json
import pandas as pd


def load_papineau_role_nouns(
        data_path="data/papineau_role_nouns.json"):
    with open(data_path, "r") as f:
        role_nouns = json.load(f)
    
    result = []
    for item in role_nouns:
        result.extend(item)
    return set(result)


def identify_bartl_role_nouns():
    # Downloaded from: https://github.com/marionbartl/affixed_words/blob/main/words/replacements.csv
    role_noun_list = pd.read_csv("data/bartl-showgirl-performer-replacements.csv")
    role_noun_list = role_noun_list.loc[role_noun_list["frequency"] >= 1]

    papineau_role_nouns = load_papineau_role_nouns()

    df_dict = collections.defaultdict(list)
    for _, df in role_noun_list.groupby("replacement"):

        # Want forms with at masc and fem variants
        masc_df = df[df["category"].isin({"man", "boy"})]
        fem_df = df[df["category"].isin({"woman", "girl"})]
        if len(masc_df) == 0:
            continue
        if len(fem_df) == 0:
            continue

        words = set(df["word"])
        replacement = list(df["replacement"])[0]
        most_frequent_masc = masc_df.loc[masc_df['frequency'].idxmax()]["word"]
        most_frequent_fem = fem_df.loc[fem_df['frequency'].idxmax()]["word"]

        df_dict["gendered_variants"].append(words)
        df_dict["neutral"].append(replacement)
        df_dict["masculine"].append(most_frequent_masc)
        df_dict["feminine"].append(most_frequent_fem)
        df_dict["n_gendered_variants"].append(len(words))

        # if words or replacement overlap with papineau words, add a column
        if any([word in papineau_role_nouns for word in words]) or replacement in papineau_role_nouns:
            df_dict["papineau_overlap"].append("True")
        else:
            df_dict["papineau_overlap"].append("False")

    result_df = pd.DataFrame(df_dict)
    result_df.to_csv("data/bartl-showgirl-performer-selected_role_noun_sets.csv")


if __name__ == "__main__":
    identify_bartl_role_nouns()