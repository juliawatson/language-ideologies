import part_2_query_t5
import torch

model, tokenizer, device = part_2_query_t5.set_up_model(imported=True)
example_data = {
    "multi-end": ("You are choosing between \"flight attendant,\" \"steward,\" and \"stewardess.\" Of these words, the word most likely to complete the sentence \"Finley is a ____.\" is [FORM]", "flight attendant"), 
    "single-end": ("You are choosing what pronoun to use. The word most likely to complete the sentence \"Finley probably walks ten miles by ____ everyday.\" is [FORM]", "themselves"), 
    "multi-middle": ("Alex is a [FORM].", "congressperson"), 
    "single-middle": ("Finley probably walks ten miles by [FORM] everyday.", "themselves"),
    "variant-starts-with-underscore-middle": ("Cameron is a [FORM].", "maniac"),
    "variant-starts-with-underscore-end": ("The word most likely to complete the sentence \"Cameron is a ____.\" is [FORM]", "maniac")
}

def get_output_labels_test_helper(sentence, variant):
    input_tokens = part_2_query_t5.get_input_tokens(sentence, variant)      # first function validated by tests 1-5
    return part_2_query_t5.get_output_labels(input_tokens, variant)
    

## TESTING FUNCTION get_input_tokens ##
def test_1_multi_token_var_at_end_get_input_tokens():
    input_sentence, variant = example_data["multi-end"]
    actual = part_2_query_t5.get_input_tokens(input_sentence, variant)[0]
    actual_tokens = tokenizer.convert_ids_to_tokens(actual)
    expected_tokens = ['▁You', '▁are', '▁choosing', '▁between', '▁"', 'f', 'light', '▁attendant', 
                       ',"', '▁"', 'ste', 'ward', ',"', '▁and', '▁"', 'ste', 'ward', 'e', 's', 's', '."', 
                       '▁Of', '▁these', '▁words', ',', '▁the', '▁word', '▁most', '▁likely', '▁to', 
                       '▁complete', '▁the', '▁sentence', '▁"', 'F', 'in', 'ley', '▁is', '▁', 'a', '▁', 
                       '_', '_', '_', '_', '."', '▁is', '<extra_id_0>', '</s>']
    assert actual_tokens == expected_tokens


def test_2_multi_token_var_at_middle_get_input_tokens():
    input_sentence, variant = example_data["multi-middle"]
    actual = part_2_query_t5.get_input_tokens(input_sentence, variant)[0]
    actual_tokens = tokenizer.convert_ids_to_tokens(actual)
    expected_tokens = ['▁Alex', '▁is', '▁', 'a', '<extra_id_0>', '.', '</s>']
    assert actual_tokens == expected_tokens


def test_3_single_token_var_at_end_get_input_tokens():
    input_sentence, variant = example_data["single-end"]
    actual = part_2_query_t5.get_input_tokens(input_sentence, variant)[0]
    actual_tokens = tokenizer.convert_ids_to_tokens(actual)
    expected_tokens = ['▁You', '▁are', '▁choosing', '▁what', '▁pro', 'nou', 'n', 
                       '▁to', '▁use', '.', '▁The', '▁word', '▁most', '▁likely', 
                       '▁to', '▁complete', '▁the', '▁sentence', '▁"', 'F', 'in', 'ley', 
                       '▁probably', '▁walks', '▁', 'ten', '▁miles', '▁by', '▁', '_', 
                       '_', '_', '_', '▁everyday', '."', '▁is', '<extra_id_0>', '</s>']
    assert actual_tokens == expected_tokens


def test_4_single_token_var_at_middle_get_input_tokens():
    input_sentence, variant = example_data["single-middle"]
    actual = part_2_query_t5.get_input_tokens(input_sentence, variant)[0]
    actual_tokens = tokenizer.convert_ids_to_tokens(actual)
    expected_tokens =  ['▁Fin', 'ley', '▁probably', '▁walks', '▁', 'ten', '▁miles', 
                        '▁by', '<extra_id_0>', '▁everyday', '.', '</s>']
    assert actual_tokens == expected_tokens

def test_5_role_noun_var_starts_with_underscore_token():
    # "maniac" is tokenized as ('▁', 'mania', 'c')
    input_sentence, variant = example_data["variant-starts-with-underscore-middle"]
    actual = part_2_query_t5.get_input_tokens(input_sentence, variant)[0]
    actual_tokens = tokenizer.convert_ids_to_tokens(actual)
    expected_tokens = ['▁Cameron', '▁is', '▁', 'a', '<extra_id_0>', '.', '</s>']
    assert actual_tokens == expected_tokens


def test_5b_role_noun_var_starts_with_underscore_token_end():
    # "maniac" is tokenized as ('▁', 'mania', 'c')
    input_sentence, variant = example_data["variant-starts-with-underscore-end"]
    actual = part_2_query_t5.get_input_tokens(input_sentence, variant)[0]
    actual_tokens = tokenizer.convert_ids_to_tokens(actual)
    expected_tokens = [
        '▁The', '▁word', '▁most', '▁likely', '▁to', '▁complete', '▁the', '▁sentence', '▁"',
        'C', 'a', 'mer', 'on', '▁is', '▁', 'a', '▁', '_', '_', '_', '_', '."', '▁is', 
        '<extra_id_0>', '</s>']
    assert actual_tokens == expected_tokens


## TESTING FUNCTION get_output_labels ##
def test_6_multi_token_var_at_end_get_output_labels():
    input_sentence, variant = example_data["multi-end"]
    actual_label, actual_exclusion = get_output_labels_test_helper(input_sentence, variant)
    actual_label_tokens = tokenizer.convert_ids_to_tokens(actual_label[0])
    expected_label_tokens =  ['<extra_id_0>', '▁flight', '▁attendant', '</s>']
    expected_exclusion = {0, len(expected_label_tokens) - 1}        # sentinel 0 and </s>
    
    assert (actual_label_tokens == expected_label_tokens) and (actual_exclusion == expected_exclusion)


def test_7_multi_token_var_at_middle_get_output_labels():
    input_sentence, variant = example_data["multi-middle"]
    actual_label, actual_exclusion = get_output_labels_test_helper(input_sentence, variant)
    actual_label_tokens = tokenizer.convert_ids_to_tokens(actual_label[0])
    expected_label_tokens = ['<extra_id_0>', '▁congress', 'person', '<extra_id_1>', '</s>']
    expected_exclusion = {0, len(expected_label_tokens) - 1, len(expected_label_tokens) - 2}        # sentinel 0, 1 and </s>
    
    assert (actual_label_tokens == expected_label_tokens) and (actual_exclusion == expected_exclusion)


def test_8_single_token_var_at_end_get_output_labels():
    input_sentence, variant = example_data["single-end"]
    actual_label, actual_exclusion = get_output_labels_test_helper(input_sentence, variant)
    actual_label_tokens = tokenizer.convert_ids_to_tokens(actual_label[0])
    
    expected_label_tokens = ['<extra_id_0>', '▁themselves', '</s>']
    expected_exclusion = {0, len(expected_label_tokens) - 1}        # sentinel 0 and </s>
    
    assert (actual_label_tokens == expected_label_tokens) and (actual_exclusion == expected_exclusion)


def test_9_single_token_var_at_middle_get_output_labels():
    input_sentence, variant = example_data["single-middle"]
    actual_label, actual_exclusion = get_output_labels_test_helper(input_sentence, variant)
    actual_label_tokens = tokenizer.convert_ids_to_tokens(actual_label[0])
    
    expected_label_tokens = ['<extra_id_0>', '▁themselves', '<extra_id_1>', '</s>']
    expected_exclusion = {0, len(expected_label_tokens) - 1, len(expected_label_tokens) - 2}        # sentinel 0, 1 and </s>
    
    assert (actual_label_tokens == expected_label_tokens) and (actual_exclusion == expected_exclusion)


def test_10_role_noun_var_starts_with_underscore_token():
    input_sentence, variant = example_data["variant-starts-with-underscore-middle"]
    actual_label, actual_exclusion = get_output_labels_test_helper(input_sentence, variant)
    actual_label_tokens = tokenizer.convert_ids_to_tokens(actual_label[0])
    expected_label_tokens =  ['<extra_id_0>', '▁', 'mania', 'c', '<extra_id_1>', '</s>']
    expected_exclusion = {0, len(expected_label_tokens) - 1, len(expected_label_tokens) - 2}        # sentinel 0, 1 and </s>
    
    assert (actual_label_tokens == expected_label_tokens) and (actual_exclusion == expected_exclusion)


def test_10b_role_noun_var_starts_with_underscore_token():
    input_sentence, variant = example_data["variant-starts-with-underscore-end"]
    actual_label, actual_exclusion = get_output_labels_test_helper(input_sentence, variant)
    actual_label_tokens = tokenizer.convert_ids_to_tokens(actual_label[0])
    expected_label_tokens =  ['<extra_id_0>', '▁', 'mania', 'c', '</s>']
    expected_exclusion = {0, len(expected_label_tokens) - 1}
    
    assert (actual_label_tokens == expected_label_tokens) and (actual_exclusion == expected_exclusion)


## TESTING FUNCTION compute_probabilities ##
def test_11_compute_probabilities_sample_small_logit():
    # each tensor indicates probs over ['</s>', 'person', 'hi', '▁congress', '<extra_id_0>', '<extra_id_1>']
    # only the target log probs are -1. Other probs are all -5.
    logprobs = [torch.tensor([-5., -5., -5., -5., -1., -5.]),  # 0th token: '<extra_id_0>'
                        torch.tensor([-5., -5., -5., -1., -5., -5.]),  # 1st token: '▁congress'
                        torch.tensor([-5., -1., -5., -5., -5., -5.]),  # 2nd token: 'person'
                        torch.tensor([-5., -5., -5., -5., -5., -1.]),  # 3rd token: '<extra_id_1>'
                        torch.tensor([-1., -5., -5., -5., -5., -5.])]  # 4th token: '</s>'

    # ['<extra_id_0>', '▁congress', 'person', '<extra_id_1>', '</s>']
    output_labels = torch.tensor([[4, 3, 1, 5, 0]], dtype=torch.int64)

    exclusions = {0, 3, 4}

    expected = -2.
    actual = part_2_query_t5.compute_probabilities(logprobs, output_labels, exclusions)

    assert expected == actual