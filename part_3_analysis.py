import collections
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import pingouin as pg
import scipy.stats as st
import seaborn as sns

from common import load_config
import constants


DPI = 100  # 500


sns.set_context("talk", font_scale=0.9)


def get_p_reform(logprobs_dict, variant_set):
    """Returns the probability of a reform variant, given one of the variants is used.

    logprobs_dict is a dict mapping variant to its (unnormalized) log probabilities.
    """
    for curr_logprob in logprobs_dict.values():
        assert np.exp(curr_logprob) > 0.

    denominator = np.sum([np.exp(curr_logprob) for curr_logprob in logprobs_dict.values()])
    normalized_probs_dict = {k: np.exp(v) / denominator for k, v in logprobs_dict.items()}
    assert 0.99 < np.sum(list(normalized_probs_dict.values())) <= 1.01, np.sum(list(normalized_probs_dict.values()))
    result = 0
    for variant, p_variant in normalized_probs_dict.items():
        if variant in variant_set:
            result += p_variant
    return result


def filter_on_plot_conditions(df, config):
    if "plot_contexts" in config:
        df = df.loc[[context in config["plot_contexts"] for context in df["context"]]]
    if "plot_ways_of_asking" in config:
        df = df.loc[[way_of_asking in config["plot_ways_of_asking"] for way_of_asking in df["way_of_asking"]]]
    return df


def make_summary_table(df, config, config_path, model_name):
    output_path = config_path.replace("config.json", f"{model_name}/p_reform_summary_table.csv")
    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=["way_of_asking"] + list(config["contexts"]))
        csv_writer.writeheader()
        for way_of_asking in config["ways_of_asking"]:
            curr_row = {"way_of_asking": way_of_asking}
            for context in config["contexts"]:
                prompt_p_variants = df[
                    (df["way_of_asking"] == way_of_asking) & 
                    (df["context"] == context)
                ]["p(reform)"]
                curr_row[context] = np.mean(prompt_p_variants)
            csv_writer.writerow(curr_row)

    # Focus on plot conditions in summary tables that average over ways of asking / contexts
    plot_conditions_df = filter_on_plot_conditions(df, config)

    way_of_asking_path = config_path.replace("config.json", f"{model_name}/p_reform_summary_table_way_of_asking.csv")
    df_way_of_asking = plot_conditions_df[["way_of_asking", "p(reform)"]].groupby("way_of_asking").mean()
    df_way_of_asking.to_csv(way_of_asking_path)

    context_path = config_path.replace("config.json", f"{model_name}/p_reform_summary_table_context.csv")
    df_context = plot_conditions_df[["context", "p(reform)"]].groupby("context").mean()

    df_context.to_csv(context_path)


def make_plot(df, config, config_path, model_name, default_color="#AC98E0"):
    df = filter_on_plot_conditions(df, config)

    # Plotting for experiment 1
    if "experiment1" in config_path:
        domain = config_path.split("/")[2]

        for factor in ["context", "way_of_asking"]:
            # plot_path = config_path.replace("config.json", f"{MODEL_NAME}/p_reform_barplot_{factor}.png")
            plot_path = f"appendix-figures/exp-1/p_reform_barplot_context_{model_name}_{domain}_{factor}.png"
            if factor == "context":
                order = config["plot_contexts"] if "plot_contexts" in config else config["contexts"]
                label_dict = {
                    "choices-all-terms": "choices",
                    "choices-pronoun": "choices",
                    "ideology-declaration": "ideology\ndeclaration",
                    "individual-declaration": "individual\ndeclaration",
                    "null_context": "null"
                }
            elif factor == "way_of_asking":
                order = config["plot_ways_of_asking"] if "plot_ways_of_asking" in config else config["ways_of_asking"]
                label_fn = lambda x: x
                label_dict = {
                    "likely_complete": "likely\ncomplete",
                    "best_complete": "best\ncomplete",
                    "likely_refer": "likely\nrefer",
                    "best_refer": "best\nrefer",
                }
                df = df.rename(columns={"way_of_asking": "way of asking"})
                factor = "way of asking"
            label_fn = lambda x: label_dict.get(x, x)
            plt.figure()
            sns.barplot(df, x=factor, y="p(reform)", color=default_color, order=order, formatter=label_fn)
            plt.ylim(0, 1)
            plt.xlabel("")
            plt.title(model_name, fontsize=26)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=DPI)

    # Plotting for experiment 2
    elif "experiment2" in config_path:
        sns.set_context("talk", font_scale=0.8)

        default_color = "#72D6B7"
        domain = config_path.split("/")[2]
        # plot_path = config_path.replace("config.json", f"{MODEL_NAME}/p_reform_barplot_context.png")
        plot_path = f"appendix-figures/exp-2/p_reform_barplot_context_{model_name}_{domain}.png"

        plot_order = df.groupby('context')['p(reform)'].mean().sort_values(ascending=True).index.values
        colors = [
            config["plotting"]["colors"].get(condition, default_color) for condition in plot_order]

        plt.figure(figsize=(8.5, 6))
        # plt.figure(figsize=(8.5, 4.8))
        ax = sns.barplot(df, x="context", y="p(reform)", palette=colors, order=plot_order, saturation=1)  # , hue="context", legend=False)
        plt.xticks(rotation=62, ha="right", rotation_mode="anchor")
        # plt.xticks(rotation=90)

        # positive_qualities_patch = mpatches.Patch(color=default_color, label='positive qualities')
        # prog_patch = mpatches.Patch(color='#695598', label='progressive')
        # prog_stance_patch = mpatches.Patch(color='#AC98E1', label='prog stance')
        # cons_patch = mpatches.Patch(color='#E46D00', label='conservative')
        # cons_stance_patch = mpatches.Patch(color='#FF9E45', label='cons stance')

        plt.plot([],[], marker="o", ms=12, ls="", color='#695598', label='prog')
        plt.plot([],[], marker="^", ms=12, ls="", color='#AC98E1', label='prog-stance')
        plt.plot([],[], marker="o", ms=12, ls="", color='#F07300', label='cons')
        plt.plot([],[], marker="^", ms=12, ls="", color='#FF9E45', label='cons-stance')
        plt.plot([],[], marker="D", ms=10, ls="", color=default_color, label='positive-metaling')
        
        plt.legend(
            #handles=[prog_patch, prog_stance_patch, cons_patch, cons_stance_patch, positive_qualities_patch],
            bbox_to_anchor=(0, 1.05, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            ncol=3
        )
        plt.xlabel("")
        plt.ylim(0, 1)
        plt.tight_layout()
        ax.margins(x=0.01) 
        ax.figure.set_size_inches(8.5, 6)
        plt.savefig(plot_path, bbox_inches="tight", dpi=DPI)
        sns.set_context("talk", font_scale=0.9)
        

def experiment_2_analysis(df, config, config_path, model_name):
    # Note: this analysis raises warnings in some cases (RuntimeWarning: divide by zero encountered in scalar divide)
    # This happens when p-values are too small to be represented in python (and so are 0.0).
    # When pg.ttest() is called, it computes several stats, including BF10 which it computes as:
    #   bf10 = 1 / ((1 + t**2 / df) ** (-(df + 1) / 2) / integr)
    # since t=0.0, this divides by 0, and so causes an error.
    # Because we aren't using BF10 in our analysis, we can ignore the error.
    exp2_df = get_conditions_by_items_df(df, config)

    # Phase 1: Do prompts with progressive groups/stances have higher p(reform), compared to 
    # conservative groups/stances?
    test1_path = config_path.replace("config.json", f"{model_name}/compare_political_groups.csv")
    test1_df = test_compare_political_groups(exp2_df)
    test1_df.to_csv(test1_path)

    # Phase 2: Do the progressive stances pattern like progressive groups? And do the 
    # conservative stances pattern like conservative groups?
    test2_path = config_path.replace("config.json", f"{model_name}/compare_stance_to_political_groups.csv")
    test2_df = test_compare_stances_to_political_groups(exp2_df)
    test2_df.to_csv(test2_path)

    # Phase 3: Do the metalinguistic qualities pattern more like progressive groups/stances?
    test3_path = config_path.replace("config.json", f"{model_name}/compare_metalinguistic_to_political_groups.csv")
    test3_df = test_compare_metalinguistic_to_political_groups(exp2_df)
    test3_df.to_csv(test3_path)


def add_p_reform_by_item_column(result_df, context_df, column_label):
    context_df = context_df.rename(columns={"p(reform)": column_label})
    context_df = context_df[["name", "item", column_label]]
    context_df = context_df.groupby(["name", "item"]).mean()
    return result_df.merge(context_df, on=["name", "item"])


def get_conditions_by_items_df(df, config):
    result_df = df[["name", "item"]].drop_duplicates()

    # Add p(reform) column for each prompt
    for curr_prompt_label in set(df["context"]):
        result_df = add_p_reform_by_item_column(
            result_df, 
            df[df["context"] == curr_prompt_label],
            curr_prompt_label)

    # Add averaged p(reform) column for each political prompt group
    for prompt_type in ["political", "stances"]:
        for prompt_group_label, prompt_group_list in config["analysis_groups"][prompt_type].items():
            result_df = add_p_reform_by_item_column(
                result_df, 
                df[df["context"].isin(prompt_group_list)],
                prompt_group_label)

    # Add averaged p(reform) column for the positive metalinguistic prompt group
    result_df = add_p_reform_by_item_column(
        result_df, 
        df[df["context"].isin(config["analysis_groups"]["positive-metalinguistic"])],
        "positive-metalinguistic")

    return result_df


def test_compare_political_groups(exp2_df):
    # 1(a) compare progressive-group vs. conservative-group
    test_result_1a = t_test_helper_exp2(
        exp2_df, "progressive-group", "conservative-group", alternative="greater")
    # test_result_1 = pg.ttest(
    #     exp2_df["progressive-group"], exp2_df["conservative-group"], paired=True)
    # test_result_1["label"] = "political-group"
    # test_result_1["progressive-mean"] = np.mean(exp2_df["progressive-group"])
    # test_result_1["conservative-mean"] = np.mean(exp2_df["conservative-group"])

    # 1(b) compare progressive-stance-group vs. conservative-stance-group
    test_result_1b = t_test_helper_exp2(
        exp2_df, "progressive-stance-group", "conservative-stance-group", alternative="greater")
    # test_result_2 = pg.ttest(
    #     exp2_df["progressive-stance-group"], exp2_df["conservative-stance-group"], paired=True)
    # test_result_2["label"] = "political-stance-group"
    # test_result_2["progressive-mean"] = np.mean(exp2_df["progressive-stance-group"])
    # test_result_2["conservative-mean"] = np.mean(exp2_df["conservative-stance-group"])

    result = pd.concat([test_result_1a, test_result_1b]).reset_index()
    result = result[["label"] + [column for column in result.columns if column != "label"]]
    return result


def t_test_helper_exp2(df, a, b, alternative="two-sided"):
    # Helper for Exp2 analyses:
    # Conduct a test comparing a to b
    # Here, df is exp2df, and a and b are column labels
    test_result = pg.ttest(df[a], df[b], paired=True, alternative=alternative)
    if alternative == "two-sided":
        test_result["label"] = f"{a} != {b}?"
    elif alternative == "greater":
        test_result["label"] = f"{a} > {b}?"
    elif alternative == "less":
        test_result["label"] = f"{a} < {b}?"

    test_result["means-difference"] = np.mean(df[a]) - np.mean(df[b])
    test_result["means-difference-abs"] = np.abs(np.mean(df[a]) - np.mean(df[b]))

    test_result["a"] = a
    test_result["b"] = b
    test_result["a-mean"] = np.mean(df[a])
    test_result["b-mean"] = np.mean(df[b])
    
    return test_result


def pivot_test_helper_exp2(df, pivot, a, b):
    # Runs t-test assessing if df[pivot] is closer to df[a] or df[b]
    pivot_to_a = np.abs(df[pivot] - df[a])
    pivot_to_b = np.abs(df[pivot] - df[b])
    test_result = pg.ttest(pivot_to_a, pivot_to_b, paired=True, alternative="two-sided")

    test_result["label"] = f"{pivot} closer to {a} or {b}?"
    test_result["means-difference"] = np.mean(pivot_to_a) - np.mean(pivot_to_b)
    test_result["means-difference-abs"] = np.abs(np.mean(pivot_to_a) - np.mean(pivot_to_b))

    test_result["pivot"] = pivot
    test_result["a"] = a
    test_result["b"] = b

    test_result["pivot-mean"] = np.mean(df[pivot])
    test_result["a-mean"] = np.mean(df[a])
    test_result["b-mean"] = np.mean(df[b])
    return test_result


def test_compare_stances_to_political_groups(exp2_df):
    # 2(a): Are progressive stances closer to progressive groups or conservative groups?
    test_result_2a = pivot_test_helper_exp2(
        exp2_df, "progressive-stance-group", "progressive-group", "conservative-group")
    
    #2(b): Are conservative stances closer to conservative groups or progressive groups?
    test_result_2b = pivot_test_helper_exp2(
        exp2_df, "conservative-stance-group", "progressive-group", "conservative-group")

    result = pd.concat([test_result_2a, test_result_2b]).reset_index()
    result = result[["label"] + [column for column in result.columns if column != "label"]]
    return result


def test_compare_metalinguistic_to_political_groups(exp2_df):
    # 3(a): Are metalinguistic prompts closer to progressive groups or conservative groups?
    test_result_3a = pivot_test_helper_exp2(
        exp2_df, "positive-metalinguistic", "progressive-group", "conservative-group")

    # 3(b): Are metalinguistic prompts closer to progressive stances or conservative stances?
    test_result_3b = pivot_test_helper_exp2(
        exp2_df, "positive-metalinguistic", "progressive-stance-group", "conservative-stance-group")

    result = pd.concat([test_result_3a, test_result_3b]).reset_index()
    result = result[["label"] + [column for column in result.columns if column != "label"]]
    return result


def load_comparison_conditions(df, config, model_name):
    # Load data for comparison conditions from past experiments, and append them to df
    if "other_comparison_conditions" in config:
        for comparison_condition in config["other_comparison_conditions"]:
            comparison_logprobs_path = f"{comparison_condition['experiment_dir']}/{model_name}/logprobs.csv"
            if not os.path.exists(comparison_logprobs_path):
                print(f"Skipping {comparison_condition['experiment_dir']} for model={model_name}")
                continue
            comparison_df = pd.read_csv(comparison_logprobs_path)

            # Filter to select a subset of conditions
            if "context" in comparison_condition or "way_of_asking" in comparison_condition:
                if isinstance(comparison_condition["context"], str):
                    comparison_condition_contexts = [comparison_condition["context"]]
                else:
                    comparison_condition_contexts = comparison_condition["context"]
                
                if isinstance(comparison_condition["way_of_asking"], str):
                    comparison_condition_ways_of_asking = [comparison_condition["way_of_asking"]]
                else:
                    comparison_condition_ways_of_asking = comparison_condition["way_of_asking"]

                comparison_df = comparison_df[
                    (comparison_df["context"].isin(comparison_condition_contexts)) & 
                    (comparison_df["way_of_asking"].isin(comparison_condition_ways_of_asking))]
                
                if "context_rename" in comparison_condition:
                    assert isinstance(comparison_condition["context"], str)
                    comparison_df["context"] = comparison_condition["context_rename"]
                if "way_of_asking_rename" in comparison_condition:
                    assert isinstance(comparison_condition["way_of_asking"], str)
                    comparison_df["way_of_asking"] = comparison_condition["way_of_asking_rename"]
                
            df = pd.concat([df, comparison_df]).reset_index(drop=True)
    
    return df


def average_choices_all_terms(df):
    if "choices-all-terms" in set(df["context"]):
        all_terms = df.loc[df["context"] == "choices-all-terms"]
        all_terms = all_terms.groupby(["name", "item", "context", "way_of_asking"]).mean()
        all_terms = all_terms.reset_index()
        other_contexts = df.loc[df["context"] != "choices-all-terms"]
        df = pd.concat([all_terms, other_contexts]).reset_index(drop=True)
    return df


def remove_excluded_items(df, config):
    if "exclude_items" in config:
        return df[~df["item"].isin(config["exclude_items"])]
    return df


def main(config_path, model_name):
    # Load the config
    config = load_config(config_path)

    # Make model dir if it does not exist
    model_dir = config_path.replace("config.json", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load logprobs associated with config
    logprobs_path = f"{model_dir}/logprobs.csv"
    if os.path.exists(logprobs_path):
        df = pd.read_csv(logprobs_path)
    else:
        df = pd.DataFrame()

    # Load data for comparison conditions from past experiments (if applicable), 
    # and append them to df
    df = load_comparison_conditions(df, config, model_name)

    # remove any items to exclude
    df = remove_excluded_items(df, config)

    # Compute p(reform)
    df["p(reform)"] = [
        get_p_reform(eval(item), config["reform_variants"]) 
        for item in df["logprobs_dict"]]
    df = df[["name", "item", "context", "way_of_asking", "p(reform)"]]

    # average choices-all-terms per item (since there's 6 of them)
    df = average_choices_all_terms(df)

    # Check that we have the correct number of data points
    if isinstance(config["expected_n_data_points"], int):
        expected_n_data_points = config["expected_n_data_points"]
    elif model_name in config["expected_n_data_points"]:
        expected_n_data_points = config["expected_n_data_points"][model_name]
    else:
        expected_n_data_points = config["expected_n_data_points"]["default"]
    assert len(df) == expected_n_data_points

    # Save the data frame for running regressions later
    df.to_csv(config_path.replace("config.json", f"{model_name}/regression_input_data.csv"))
    
    make_summary_table(df, config, config_path, model_name)
    make_plot(df, config, config_path, model_name)

    # We do experiment 2 analys in this script. Experiment 1 analysis is done in R 
    # (since it requires regression tests)
    if "experiment2" in config_path and "pilot" not in config_path:
        experiment_2_analysis(df, config, config_path, model_name=model_name)


if __name__ == "__main__":
    # main("analyses/experiment1/role-nouns-full/config.json")
    # main("analyses/experiment1/singular-pronouns-full/config.json")

    # main("analyses/experiment2/role-nouns-full/config.json")
    # main("analyses/experiment2/singular-pronouns-full/config.json")

    # main("analyses/experiment1/role-nouns-expanded/config.json")
    # main("analyses/experiment2/role-nouns-expanded/config.json")

    models = [
        "text-curie-001", "text-davinci-002", "text-davinci-003",
        "flan-t5-small", "flan-t5-large", "flan-t5-xl",
        "llama-2-7B", "llama-3-8B", "llama-3.1-8B"]
    for model_name in models:
        print(f"model_name={model_name}")
        main("analyses/experiment1/singular-pronouns-full/config.json", model_name=model_name)
        main("analyses/experiment1/role-nouns-full-minus-anchor-flight-attendant/config.json", model_name=model_name)
        main("analyses/experiment1/role-nouns-full-minus-anchor-flight-attendant-plus-expanded/config.json", model_name=model_name)
        
        main("analyses/experiment2/singular-pronouns-full/config.json", model_name=model_name)
        main("analyses/experiment2/role-nouns-full-minus-anchor-flight-attendant/config.json", model_name=model_name)
        main("analyses/experiment2/role-nouns-full-minus-anchor-flight-attendant-plus-expanded/config.json", model_name=model_name)
