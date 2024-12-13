import collections
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from part_3_analysis import *


sns.set_context("poster", font_scale=0.6)
# sns.set_context("poster")


analysis_to_label = {
    "progressive-stance-group closer to progressive-group or conservative-group?": "prog-stance\ncloser to\nprog or cons",
    "conservative-stance-group closer to progressive-group or conservative-group?": "cons-stance\ncloser to\nprog or cons",
    "positive-metalinguistic closer to progressive-group or conservative-group?": "positive-metaling\ncloser to\nprog or cons",
    "positive-metalinguistic closer to progressive-stance-group or conservative-stance-group?": "positive-metaling\ncloser to\nprog-stance or cons-stance"
}

marker_dict = {
    'conservative-group': "o",
    'conservative-stance-group': "^",
    'positive-metalinguistic': "D",
    'progressive-group': "o",
    'progressive-stance-group': "^"
}
ORANGE = "#FF9E45"
DARK_ORANGE = "#F07300"
PURPLE = "#AC98E0"
DARK_PURPLE = "#695598"
GREEN = "#72D6B7"
color_dict = {
    'conservative-group': DARK_ORANGE,
    'conservative-stance-group': ORANGE,
    'positive-metalinguistic': GREEN,
    'progressive-group': DARK_PURPLE,
    'progressive-stance-group': PURPLE
}


def process_results_single_model(results_path, model_name, n_models, alpha=0.05):
    # Load data
    curr_df = pd.read_csv(results_path)

    value_vars = ["pivot", "a", "b"]
    value_vars_means = ["pivot-mean", "a-mean", "b-mean"]
    condition_results_df = curr_df.melt(id_vars = ["label"], value_vars = value_vars)
    condition_results_df = condition_results_df.rename(columns={"variable": "pivot-test-label", "value": "condition"})
    cond_values_df = curr_df.melt(id_vars = ["label"], value_vars = value_vars_means)

    assert (cond_values_df["label"] == condition_results_df["label"]).all()
    assert (condition_results_df["pivot-test-label"] == [item.split("-")[0] for item in cond_values_df["variable"]]).all()
    
    condition_results_df["p(reform)"] = cond_values_df["value"]
    condition_results_df["model"] = model_name
    condition_results_df = condition_results_df[["model", "condition", "p(reform)"]].drop_duplicates()

    # Build a dataframe that keeps track of what to plot per analysis
    n_comparisons = len(curr_df) * n_models
    print(f"Bonferroni correction for {n_comparisons} - model results")
    p_val_threshold = alpha / (len(curr_df) * n_models)

    plotting_df_dict = collections.defaultdict(list)
    for _, row in curr_df.iterrows():

        significance_lines = []
        if row["p-val"] < p_val_threshold:
            # row["means-difference"] is mean(dist_to_a) - mean(dist_to_b)
            # when row["means-difference"] < 0, dist_to_a is smaller, so "a" is closer
            closer = row["a"] if row["means-difference"] < 0 else row["b"]
            closer2 = row["a"] if row["T"] < 0 else row["b"]
            assert closer == closer2
            significance_lines.append((row["pivot"], closer))
        
        plotting_df_dict["analysis"].append(analysis_to_label[row["label"]])
        plotting_df_dict["model"].append(model_name)
        plotting_df_dict["conditions"].append({row["a"], row["b"], row["pivot"]})
        plotting_df_dict["significance_lines"].append(significance_lines)

    plotting_df = pd.DataFrame(plotting_df_dict)
    return condition_results_df, plotting_df


# def load_plotting_data(config_path, models):

#     results_df = None
#     plotting_df = None
#     for model in models:
#         # # # Stage 1: Load csv + process columns
#         # stage1_path = config_path.replace("config.json", f"{model}/compare_political_groups.csv")
#         # stage1_results_df, stage1_plotting_df = process_results_single_model(stage1_path, model, n_models=len(models))

#         # Stage 2: Load csv + process columns
#         stage2_path = config_path.replace("config.json", f"{model}/compare_stance_to_political_groups.csv")
#         stage2_results_df, stage2_plotting_df = process_results_single_model(stage2_path, model, n_models=len(models))

#         # Stage 3: Load csv + process columns
#         stage3_path = config_path.replace("config.json", f"{model}/compare_metalinguistic_to_political_groups.csv")
#         stage3_results_df, stage3_plotting_df = process_results_single_model(stage3_path, model, n_models=len(models))

#         # Combine results for stages 1-3
#         if results_df is None:
#             results_df = pd.concat([stage2_results_df, stage3_results_df]).drop_duplicates().reset_index(drop=True)
#             plotting_df = pd.concat([stage2_plotting_df, stage3_plotting_df]).reset_index(drop=True)
#         else:
#             results_df = pd.concat([results_df, stage2_results_df, stage3_results_df]).drop_duplicates().reset_index(drop=True)
#             plotting_df = pd.concat([plotting_df, stage2_plotting_df, stage3_plotting_df]).reset_index(drop=True)

#     return results_df, plotting_df


def load_plotting_data(config_path, models):

    results_df = None
    plotting_df = None
    for model in models:
        # # # Stage 1: Load csv + process columns
        # stage1_path = config_path.replace("config.json", f"{model}/compare_political_groups.csv")
        # stage1_results_df, stage1_plotting_df = process_results_single_model(stage1_path, model, n_models=len(models))

        # Stage 2: Load csv + process columns
        # stage2_path = config_path.replace("config.json", f"{model}/compare_stance_to_political_groups.csv")
        # stage2_results_df, stage2_plotting_df = process_results_single_model(stage2_path, model, n_models=len(models))

        # Stage 3: Load csv + process columns
        stage3_path = config_path.replace("config.json", f"{model}/compare_metalinguistic_to_political_groups.csv")
        stage3_results_df, stage3_plotting_df = process_results_single_model(stage3_path, model, n_models=len(models))

        # Combine results for stages 1-3
        if results_df is None:
            # results_df = pd.concat([stage2_results_df, stage3_results_df]).drop_duplicates().reset_index(drop=True)
            # plotting_df = pd.concat([stage2_plotting_df, stage3_plotting_df]).reset_index(drop=True)
            results_df = stage3_results_df
            plotting_df = stage3_plotting_df
        else:
            results_df = pd.concat([results_df, stage3_results_df]).drop_duplicates().reset_index(drop=True)
            plotting_df = pd.concat([plotting_df, stage3_plotting_df]).reset_index(drop=True)

    return results_df, plotting_df


def passes_pretest_cons_less_than_prog(config_path, model, n_models, alpha=0.05):
    pretest_path = config_path.replace("config.json", f"{model}/compare_political_groups.csv")
    curr_df = pd.read_csv(pretest_path)

    n_comparisons = len(curr_df) * n_models
    assert len(curr_df) == 2  # prog vs. cons and prog-stance vs. cons-stance
    print(f"Bonferroni correction for {n_comparisons} - pretest")
    p_val_threshold = alpha / (len(curr_df) * n_models)

    assert all([item == "greater" for item in curr_df["alternative"]])

    if all(curr_df["p-val"] < p_val_threshold):
        return True
    return False


def visualize_exp2_results(config_path, models):
    results_df, plotting_df = load_plotting_data(config_path, models)

    # analysis_groups = [
    #     "prog vs. cons", "prog-stance vs. cons-stance", 
    #     "stances vs. ideologies", "positive metalinguistic vs. stances and ideologies"]
    analysis_groups = set(plotting_df["analysis"])
    print(analysis_groups)

    # stage1_analyses = {"prog vs. cons", "prog-stance vs. cons-stance"}
    
    # analysis_yticks, analysis_tick_labels = [], []
    for analysis_label in analysis_groups:
        # analysis_yticks.append(curr_y - 2)
        # analysis_tick_labels.append(analysis_label.replace(" vs. ", "\nvs.\n"))

        if "singular-pronouns" in config_path:
            # figsize = (5.5, 3)
            # figsize = (4, 3.4)
            # figsize = (6, 6)
            figsize = (6, 5)
        else:
            # figsize = (5.5, 3.4)
            # figsize = (4, 3.4)
            # figsize = (6, 6)
            figsize = (6, 5)
        fig, ax1 = plt.subplots(1,1, figsize=figsize)
        model_yticks, model_tick_labels = [], []
        curr_y = 0


        for model in models:

            model_yticks.append(curr_y)
            model_tick_labels.append(model)

            if not passes_pretest_cons_less_than_prog(config_path, model, n_models=len(models)):
                print(f"Pretest failed for model={model} and config_path={config_path}")
                curr_y -= 1
                continue
            
            curr_plotting_df = plotting_df[(plotting_df["analysis"] == analysis_label) & (plotting_df["model"] == model)]
            assert len(curr_plotting_df) == 1
            curr_plotting_row = curr_plotting_df.reset_index(drop=True).loc[0]

            for significance_line in curr_plotting_row["significance_lines"]:
                assert len(significance_line) == 2
                point1 = lookup_p_reform(results_df, significance_line[0], model)                
                point2 = lookup_p_reform(results_df, significance_line[1], model)
                if "positive-metalinguistic" in significance_line:
                    assert len(significance_line) == 2
                    other_item = [item for item in significance_line if item != "positive-metalinguistic"][0]
                    curr_color = color_dict[other_item]
                else:
                    curr_color = "black"

                if curr_color == ORANGE:
                    curr_color = DARK_ORANGE
                plt.plot([point1, point2], [curr_y, curr_y], c=curr_color, zorder=4)

            for condition in curr_plotting_row["conditions"]:
                curr_p_reform = lookup_p_reform(results_df, condition, model)
                if condition == "positive-metalinguistic":
                    zorder=3
                else:
                    zorder = 1             
                plt.scatter([curr_p_reform], [curr_y], marker=marker_dict[condition], c=color_dict[condition], zorder=zorder, s=(plt.rcParams['lines.markersize'] ** 2) * 1.5)
            curr_y -= 1
        
        # ax1.set_yticks(analysis_yticks, analysis_tick_labels)

        # ax1.yaxis.tick_right()
        ax1.set_yticks(model_yticks, model_tick_labels)
        # ax2 = ax1.secondary_yaxis('right')
        # ax2.set_yticks(model_yticks, model_tick_labels)

        output_path = config_path.replace("config.json", f"line_visualization-{analysis_label}.png")
        output_path = output_path.replace(' ', '-').replace("\n", '-')
        plt.ylim(0.5, curr_y + 0.5)
        plt.gca().invert_yaxis()
        plt.xlabel("p(reform|context)")
        plt.margins(x=0.08)
        plt.tight_layout()
        plt.savefig(output_path, dpi=350)


def lookup_p_reform(results_df, condition, model):
    curr_condition_df = results_df[(results_df["model"] == model) & (results_df["condition"] == condition)]
    assert len(curr_condition_df) == 1
    return list(curr_condition_df["p(reform)"])[0]


if __name__ == "__main__":
    models = [
        "text-curie-001", "text-davinci-002", "text-davinci-003",
        "flan-t5-small", "flan-t5-large", "flan-t5-xl",
        "llama-2-7B", "llama-3-8B", "llama-3.1-8B"]

    visualize_exp2_results("analyses/experiment2/singular-pronouns-full/config.json", models)
    visualize_exp2_results("analyses/experiment2/role-nouns-full-minus-anchor-flight-attendant/config.json", models)
    visualize_exp2_results("analyses/experiment2/role-nouns-full-minus-anchor-flight-attendant-plus-expanded/config.json", models)
