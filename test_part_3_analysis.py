import numpy as np
import pandas as pd

import part_3_analysis


def test_average_choices_all_terms():
    df = pd.DataFrame({
        "name": ["Alex"] * 7,
        "item": ["congressperson"] * 7,
        "context": ["choices-all-terms"] * 6 + ["ideology-declaration"],
        "way_of_asking": ["best_complete"] * 7,
        "p(reform)": [0.01, 0.02, 0.03, 0.01, 0.03, 0.02, 0.60]
    })

    expected_df = pd.DataFrame({
        "name": ["Alex"] * 2,
        "item": ["congressperson"] * 2,
        "context": ["choices-all-terms", "ideology-declaration"],
        "way_of_asking": ["best_complete"] * 2,
        "p(reform)": [0.02, 0.60]
    })
    actual_df = part_3_analysis.average_choices_all_terms(df)
    
    assert ((actual_df == expected_df).all()).all()


def test_load_comparison_conditions():
    config = {
        "other_comparison_conditions": [
            {
                "experiment_dir": ".",
                "way_of_asking": "best_complete",
                "context": "ideology-declaration",
                "context_rename": "gender-inclusive"
            }
        ]
    }
    df = pd.DataFrame({
        "name": ["Alex", "Bella"] * 2,
        "item": ["[NOUN] must have stubbed [FORM] toe."] * 4,
        "context": ["natural"] * 2 + ["correct"] * 2,
        "way_of_asking": ["best_complete"] * 4,
        "logprobs_dict": [
            "{'their': -10., 'her': -1., 'his': -5.}",
            "{'their': -9., 'her': -2., 'his': -5.}",
            "{'their': -8., 'her': -3., 'his': -5.}",
            "{'their': -7., 'her': -4., 'his': -5.}",
        ]
    })

    actual = part_3_analysis.load_comparison_conditions(
        df, config, model_name="testdata")
    expected = pd.DataFrame({
        "name": ["Alex", "Bella"] * 3,
        "item": ["[NOUN] must have stubbed [FORM] toe."] * 6,
        "context": ["natural"] * 2 + ["correct"] * 2 + ["gender-inclusive"] * 2,
        "way_of_asking": ["best_complete"] * 6,
        "logprobs_dict": [
            "{'their': -10., 'her': -1., 'his': -5.}",
            "{'their': -9., 'her': -2., 'his': -5.}",
            "{'their': -8., 'her': -3., 'his': -5.}",
            "{'their': -7., 'her': -4., 'his': -5.}",
            "{'their': -1., 'her': -10., 'his': -5.}",
            "{'their': -2., 'her': -9., 'his': -5.}"
        ]
    })

    assert ((actual == expected).all()).all(), f"{actual}\n!=\n{expected}"


def test_get_p_reform():
    input_dict = {'their': -2., 'her': -9., 'his': -5.}

    actual = part_3_analysis.get_p_reform(
        input_dict, variant_set=["they", "them", "themselves", "themself", "their"])
    expected = 0.9517474055557682

    assert np.allclose(actual, expected)


def test_get_conditions_by_items_df():
    input_config = {
        "analysis_groups": {
            'political': {
                'progressive-group': ['progressive', 'liberal']},
            'stances': {
                'progressive-stance-group': [
                    'inclusive', 'avoid-misgendering', 'gender-continuum']},
            'positive-metalinguistic': ['correct', 'natural']
        }
    }
    input_df = pd.DataFrame({
        "name": ["Alex", "Wyatt"] * 7,
        "item": ["anchor", "businessperson"] * 7,
        "context": ["progressive"] * 2 + ["liberal"] * 2 + \
            ["correct"] * 2 + ["natural"] * 2 +\
            ["inclusive"] * 2 + ["avoid-misgendering"] * 2 + ["gender-continuum"] * 2,
        "p(reform)": [
            0.4, 0.5,  # progressive
            0.3, 0.4,  # liberal

            0.2, 0.3,  # correct
            0.1, 0.2,   # natural

            0.5, 0.6,  # inclusive
            0.6, 0.7,  # avoid-misgendering
            0.7, 0.8   # gender-continuum
        ]
    })

    expected = pd.DataFrame({
        "name": ["Alex", "Wyatt"],
        "item": ["anchor", "businessperson"],

        "progressive": [0.4, 0.5],
        "liberal": [0.3, 0.4],

        "inclusive": [0.5, 0.6],
        "avoid-misgendering": [0.6, 0.7],
        "gender-continuum": [0.7, 0.8],

        "correct": [0.2, 0.3],
        "natural": [0.1, 0.2],

        "progressive-group": [0.35, 0.45],
        "progressive-stance-group": [0.6, 0.7],
        "positive-metalinguistic": [0.15, 0.25]
    })
    actual = part_3_analysis.get_conditions_by_items_df(input_df, input_config)

    assert set(expected.columns) == set(actual.columns)

    for column in expected.columns:
        if column in ["name", "item"]: 
            continue
        assert np.allclose(expected[column], actual[column]), column