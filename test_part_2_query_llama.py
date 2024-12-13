import numpy as np
import torch
import part_2_query_llama


def test_get_sentence_token_logprobs():
    sentence_tokens = {
        'input_ids': torch.tensor([[4, 3, 2, 1]]),  # vocab ids
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    vocab_size = 5

    # Actual sequence has logprob of -1 at each position. Other tokens
    # have logprob of -30.
    logprobs = torch.tensor([
        [-30, -30, -30,  -1, -30],  # token "3" is probable after "4"
        [-30, -30,  -1, -30, -30],  # token "2" is probable after "3"
        [-30,  -1, -30, -30, -30],  # token "1" is probable after "2"
        [-30, -30, -30, -30, -30],  # no predictions about what's after "1" (doesn't affect result)
    ])

    actual = part_2_query_llama.get_sentence_token_logprobs(logprobs, sentence_tokens)
    expected = np.array([-1, -1, -1])
    assert np.allclose(actual, expected)


def test_compute_probability_direct():
    string_tokens = ['S', 'ally', 'Ġis', 'Ġa', 'Ġcongress', 'person', '.']
    variant = "congressperson"
    way_of_asking = "direct"
    sentence_logprobs = np.array(
        [ -5.58675957,  -5.53832388,  -3.8648417 ,  -0.8277052 ,
         -12.75976944,  -3.79423499,  -2.99609137])

    expected = np.sum(sentence_logprobs)
    actual = part_2_query_llama.compute_probability(
        sentence_logprobs, string_tokens, variant, way_of_asking)

    assert np.allclose(actual, expected)


def test_compute_probability_single_token_variant_role_noun():
    string_tokens = [
        'The', 'Ġbest', 'Ġword', 'Ġto', 'Ġcomplete', 'Ġthe', 'Ġsentence', 'Ġ"',
        'Fin', 'ley', 'Ġis', 'Ġa', 'Ġ____', '."', 'Ġis', 'Ġcongressman']
    variant = "congressman"
    way_of_asking = "best_complete"
    sentence_logprobs = np.array([
        -3.57482052,  -5.44123554,  -7.77211571,  -0.45831567,
        -8.81313038,  -0.5029338 ,  -1.45098174,  -2.04294491,
       -10.17338467,  -4.88111591,  -1.53408182,  -1.401402  ,
        -3.34153175,  -2.76285338,  -0.93264472, -15.80703545])

    expected = -15.80703545  # probability of last token
    actual = part_2_query_llama.compute_probability(
        sentence_logprobs, string_tokens, variant, way_of_asking)

    assert np.allclose(actual, expected)


def test_compute_probability_single_token_variant_singular_pronoun():
    string_tokens = [
        'The', 'Ġword', 'Ġmost', 'Ġlikely', 'Ġto', 'Ġrefer', 'Ġto', 'ĠJacob',
        'Ġin', 'Ġthe', 'Ġsentence', 'Ġ"', 'I', 'Ġhope', 'Ġthat', 'ĠJacob',
        'Ġisn', "'t", 'Ġtoo', 'Ġhard', 'Ġon', 'Ġ____', '."', 'Ġis', 'Ġhimself']
    variant = "himself"
    way_of_asking = "likely_refer"
    sentence_logprobs = np.array([
        -3.57633042,  -6.82411671,  -7.84434748,  -3.8552165 ,
        -0.34539902,  -7.01663876,  -0.0534997 , -10.58835793,
        -3.36668682,  -0.87036294,  -8.51975727,  -2.62835002,
        -3.28436708,  -4.59941196,  -1.80651152,  -0.79326242,
        -4.25203562,  -0.03447499,  -3.40593576,  -4.36110306,
        -0.17724821,  -5.03248978,  -1.92247522,  -0.94006413,
       -11.42356968])

    expected = -11.42356968  # probability of last token
    actual = part_2_query_llama.compute_probability(
        sentence_logprobs, string_tokens, variant, way_of_asking)

    assert np.allclose(actual, expected)


def test_compute_probability_multi_token_variant_role_noun():
    string_tokens = [
        'The', 'Ġbest', 'Ġword', 'Ġto', 'Ġcomplete', 'Ġthe', 'Ġsentence', 'Ġ"',
        'Fin', 'ley', 'Ġis', 'Ġa', 'Ġ____', '."', 'Ġis', 'Ġcongress', 'person']
    variant = "congressperson"
    way_of_asking = "best_complete"
    sentence_logprobs = np.array([
        -3.57482052,  -5.44123554,  -7.77211571,  -0.45831567,
        -8.81313038,  -0.5029338 ,  -1.45098174,  -2.04294491,
       -10.17338467,  -4.88111591,  -1.53408182,  -1.401402  ,
        -3.34153175,  -2.76285338,  -0.93264472, -15.54519463,
        -2.36407399])

    expected = -15.54519463 - 2.36407399  # probability of last two tokens
    actual = part_2_query_llama.compute_probability(
        sentence_logprobs, string_tokens, variant, way_of_asking)

    assert np.allclose(actual, expected)


def test_compute_probability_multi_token_variant_with_space_role_noun():
    string_tokens = [
        'The', 'Ġbest', 'Ġword', 'Ġto', 'Ġcomplete', 'Ġthe', 'Ġsentence', 'Ġ"',
        'Fin', 'ley', 'Ġis', 'Ġa', 'Ġ____', '."', 'Ġis', 'Ġcamera', 'Ġoperator']
    variant = "camera operator"
    way_of_asking = "best_complete"
    sentence_logprobs = np.array([ 
        -3.57482052,  -5.44123554,  -7.77211571,  -0.45831567,
        -8.81313038,  -0.5029338 ,  -1.45098174,  -2.04294491,
       -10.17338467,  -4.88111591,  -1.53408182,  -1.401402  ,
        -3.34153175,  -2.76285338,  -0.93264472, -12.89260674,
        -4.49586487])

    expected = -12.89260674 - 4.49586487  # probability of last two tokens
    actual = part_2_query_llama.compute_probability(
        sentence_logprobs, string_tokens, variant, way_of_asking)

    assert np.allclose(actual, expected)