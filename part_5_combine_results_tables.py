import pandas as pd


# def get_pval_str(pval):
#     # Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#     assert 0 <= pval and pval <= 1
#     if pval <= 0.001:
#         return "***"
#     if pval <= 0.01:
#         return "**"
#     if pval <= 0.05:
#         return "*"
#     if pval <= 0.1:
#         return "."
#     return "."


# def get_column(model_results_df, p_val_threshold):
#     result = []
#     for estimate, p_value in zip(model_results_df["Estimate"], model_results_df["Pr(>|z|)"]):
#         p_val_str = get_pval_str(p_value)
#         if p_value > p_val_threshold:
#             p_val_str = ""
#         result.append(f"{estimate:.2f}{p_val_str}")
#     return result


def get_column(model_results_df, p_val_threshold):
    POSITIVE_EFFECT_COLOR = "bdffea"  # blue-green
    NEGATIVE_EFFECT_COLOR = "fac0dc"  # pink
    INTERCEPT_EFFECT_COLOR = "dbd9d9" # gray
    result = []
    for estimate, p_value in zip(model_results_df["Estimate"], model_results_df["Pr(>|z|)"]):
        if result == [] and p_value < p_val_threshold:  # significant effect for Intercept -- make it gray bc no prediction
            # result.append(f"\\boldnumcolor{{{estimate:.2f}}}{{{INTERCEPT_EFFECT_COLOR}}}")
            result.append(f"\cellcolor[HTML]{{{INTERCEPT_EFFECT_COLOR}}} {estimate:.2f}")
        elif p_value < p_val_threshold:
            curr_cell_color = NEGATIVE_EFFECT_COLOR if estimate < 0 else POSITIVE_EFFECT_COLOR
            # result.append(f"\\boldnumcolor{{{estimate:.2f}}}{{{curr_cell_color}}}")
            result.append(f"\cellcolor[HTML]{{{curr_cell_color}}} {estimate:.2f}")
        else:
            result.append(f"{estimate:.2f}")
    return result


def combine_exp1_results(config_path, models, metalinguistic_only=False):
    p_val_threshold = 0.05 / len(models)  # bonferroni correction

    result_df = None
    for model in models:
        if metalinguistic_only:
            model_results_path = config_path.replace("config.json", f"{model}/regression_results-metalinguistic.csv")
        else:
            model_results_path = config_path.replace("config.json", f"{model}/regression_results.csv")
        model_results_df = pd.read_csv(model_results_path, index_col=0)

        if result_df is None:
            result_df = pd.DataFrame(index = model_results_df.index)
        else:
            assert (result_df.index == model_results_df.index).all()

        result_df[model] = get_column(model_results_df, p_val_threshold)

    if metalinguistic_only:
        output_path = config_path.replace("config.json", "overall_results-metalinguistic_only.csv")
    else:
        output_path = config_path.replace("config.json", "overall_results.csv")
    result_df.to_csv(output_path)

    output_path_latex = output_path.replace(".csv", ".tex_table")
    result_df.to_latex(output_path_latex)


def process_exp2_results_csv(results_path, model_name, n_models, alpha=0.05):
    # Load data
    curr_df = pd.read_csv(results_path, index_col="label")

    # Represent each row of data with a single string
    assert len(curr_df) == 2
    curr_p_val_threshold = alpha / (len(curr_df) * n_models)
    # p_val_strs = [get_pval_str(pval) if pval <  p_val_threshold else ""
    #               for pval in curr_df["p-val"]]
    print(f"Bonferroni correction for {len(curr_df) * n_models}")

    EFFECT_COLOR = "bdffea"  # blue-green
    def get_cell_str(means_diff, pval):
        if pval < curr_p_val_threshold:
            return f"\cellcolor[HTML]{{{EFFECT_COLOR}}} {means_diff:.2f}"
        return f"{means_diff:.2f}"

    results_strs = [get_cell_str(means_diff, pval)
                  for means_diff, pval in zip(curr_df["means-difference"], curr_df["p-val"])]

    # Return a df with the index (test labels) and a model_name column (which contains the 
    # single-string results summary for each test)
    curr_df[model_name] = results_strs
    return curr_df[[model_name]]


def combine_exp2_results(config_path, models, split=False):
    result_df = None

    prefix = "split-test-" if split else ""

    for model in models:

        # Stage 1: Load csv + process columns
        stage1_path = config_path.replace("config.json", f"{model}/compare_political_groups.csv")
        stage1_df = process_exp2_results_csv(stage1_path, model, n_models=len(models))

        # Stage 2: Load csv + process columns
        stage2_path = config_path.replace("config.json", f"{model}/{prefix}compare_stance_to_political_groups.csv")
        stage2_df = process_exp2_results_csv(stage2_path, model, n_models=len(models))

        # Stage 3: Load csv + process columns
        stage3_path = config_path.replace("config.json", f"{model}/{prefix}compare_metalinguistic_to_political_groups.csv")
        stage3_df = process_exp2_results_csv(stage3_path, model, n_models=len(models))

        # Combine results for stages 1-3
        model_df = pd.concat([stage1_df, stage2_df, stage3_df])

        # Add results for this model to the overall results df
        if result_df is None:
            result_df = pd.DataFrame(index = model_df.index)
        assert (result_df.index == model_df.index).all()
        result_df[model] = model_df[model]

        # Save results to a file
        output_path = config_path.replace("config.json", f"{prefix}overall_results.csv")
        result_df.to_csv(output_path)

        output_path_latex = output_path.replace(".csv", ".tex_table")
        result_df.to_latex(output_path_latex)



if __name__ == "__main__":
    models = [
        "text-curie-001", "text-davinci-002", "text-davinci-003",
        "flan-t5-small", "flan-t5-large", "flan-t5-xl",
        "llama-2-7B", "llama-3-8B", "llama-3.1-8B"]
    
    # combine_exp1_results("analyses/experiment1/singular-pronouns-full/config.json", models)
    # combine_exp1_results("analyses/experiment1/role-nouns-full-minus-anchor-flight-attendant/config.json", models)
    # combine_exp1_results("analyses/experiment1/role-nouns-full-minus-anchor-flight-attendant-plus-expanded/config.json", models)
    
    combine_exp2_results("analyses/experiment2/singular-pronouns-full/config.json", models)
    combine_exp2_results("analyses/experiment2/role-nouns-full-minus-anchor-flight-attendant/config.json", models)
    combine_exp2_results("analyses/experiment2/role-nouns-full-minus-anchor-flight-attendant-plus-expanded/config.json", models)


    ### 

    # combine_exp1_results("analyses/experiment1/role-nouns-full/config.json", models)
    # combine_exp1_results("analyses/experiment1/singular-pronouns-full/config.json", models)

    # combine_exp1_results("analyses/experiment1/role-nouns-full/config.json", models, metalinguistic_only=True)
    # combine_exp1_results("analyses/experiment1/singular-pronouns-full/config.json", models, metalinguistic_only=True)

    # combine_exp2_results("analyses/experiment2/role-nouns-full/config.json", models)
    # combine_exp2_results("analyses/experiment2/singular-pronouns-full/config.json", models)

    # combine_exp2_results("analyses/experiment2/role-nouns-full/config.json", models, split=True)
    # combine_exp2_results("analyses/experiment2/singular-pronouns-full/config.json", models, split=True)