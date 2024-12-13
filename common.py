import json
import pandas as pd


def load_names(names_path):
    names_df = pd.read_csv(names_path, names=["group", "name"])
    return list(names_df["name"])


def load_role_nouns(role_nouns_path):
    if role_nouns_path.endswith(".json"):
        with open(role_nouns_path) as f:
            result = json.load(f)
    else:
        assert role_nouns_path.endswith(".csv")
        role_noun_df = pd.read_csv(role_nouns_path)
        result = [
            [row["neutral"], row["masculine"], row["feminine"]]
            for _, row in role_noun_df.iterrows()
        ]
    return result


def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)

    # Load names if needed
    if "names" in config and isinstance(config["names"], str):
        config["names"] = load_names(config["names"])

    # Load role nouns from file if needed
    if config["domain"] == "role_nouns":
        if isinstance(config["role_nouns"], str):
            config["role_nouns"] = load_role_nouns(config["role_nouns"])
        else:
            assert isinstance(config["role_nouns"], list)
            role_noun_sets = []
            for role_nouns_path in config["role_nouns"]:
                role_noun_sets.extend(load_role_nouns(role_nouns_path))
            config["role_nouns"] = role_noun_sets

        if "exclude_items" in config:
            config["role_nouns"] = [
                item for item in config["role_nouns"]
                if not item[0] in config["exclude_items"]
            ]

        assert "reform_variants" not in config
        config["reform_variants"] = [item[0] for item in config["role_nouns"]]

        assert "masculine_variants" not in config
        config["masculine_variants"] = [item[1] for item in config["role_nouns"]]

        assert "feminine_variants" not in config
        config["feminine_variants"] = [item[2] for item in config["role_nouns"]]
    
    else:
        if "exclude_items" in config:
            raise ValueError(f"exclude_items not supported for domain={config['domain']}")

    return config