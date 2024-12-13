import os

import part_2_query_gpt


def test_reformat_raw_gpt():
    raw_path = "testdata/reformat_raw_input.csv"
    actual_output = "testdata/reformat_actual_output.csv"
    part_2_query_gpt.reformat_raw_gpt(raw_path, actual_output)
    actual_file_contents = open(actual_output, 'r').read()

    expected_output = "testdata/reformat_expected_output.csv"
    expected_file_contents = open(expected_output, 'r').read()
    assert expected_file_contents == actual_file_contents

    # clean-up: delete the actual_output file
    os.remove(actual_output)

