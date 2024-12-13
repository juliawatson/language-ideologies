import re

MODEL_NAMES = [
    "flan-t5-small", "flan-t5-large", "flan-t5-xl",
    # "llama-2-7B", "llama-3-8B", "llama-3.1-8B"
]

EXPERIMENTS = ["experiment1", "experiment2"]

CONFIGS = [
    "role-nouns-full", 
    #"role-nouns-expanded", 
    "singular-pronouns-full"
]


def compute_n_seconds(metadata_path):
    with open(metadata_path, "r") as f:
        metadata_str = f.read()

    assert metadata_str.count("Total seconds:") == 1, f"More than one run for metadata_path={metadata_path}"
    n_seconds_position = re.search(r"Total seconds: ", metadata_str).span()
    return float(metadata_str[n_seconds_position[1]:])




if __name__ == "__main__":
    total_n_seconds = 0
    for experiment in EXPERIMENTS:
        for config_dir in CONFIGS:
            for model_name in MODEL_NAMES:
                metadata_path = f"analyses/{experiment}/{config_dir}/{model_name}/running_metadata.txt"
                n_seconds = compute_n_seconds(metadata_path)
                total_n_seconds += n_seconds

    print(f"total_n_seconds={total_n_seconds}")

    total_n_hours = total_n_seconds / (60 * 60)
    print(f"total_n_hours={total_n_hours}")

