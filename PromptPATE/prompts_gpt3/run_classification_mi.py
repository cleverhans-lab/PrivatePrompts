import argparse
from data_utils import load_dataset_local
from utils import *
import torch.nn.functional as F
from scipy.special import softmax
import random
import numpy as np
#numpy.set_printoptions(threshold=sys.maxsize)
from sympy.utilities.iterables import multiset_permutations
def main(models, train_dataset, test_dataset, all_shots, num_seeds, subsample_test_set, api_num_log_prob, approx, use_saved_results, bs, obfuscate, must_include, start_index, end_index):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': subsample_test_set,
        'api_num_log_prob': api_num_log_prob,
        'approx': approx,
        'bs': bs
    }

    # list of all experiment parameters to run
    #all_params = []
    #for model in models:
    #    for dataset in datasets:
    #        for num_shots in all_shots:
    #            for seed in range(num_seeds):
    p = deepcopy(default_params)
    p['model'] = models[0]
    p['train_dataset'] = train_dataset
    p['test_dataset'] = test_dataset
    p['num_trails'] = num_seeds
    p['num_shots'] = all_shots[0]
    p['obfuscate'] = obfuscate
    p['must_include'] = must_include
    p['start_index'] = start_index
    p['end_index'] = end_index
    p['expr_name'] = f"teacher_{random.randint(0, 100000)}_eval_probs_{p['train_dataset']}_{p['model']}_{p['num_shots']}_shot_{num_seeds}_{p['start_index']}_{p['end_index']}_trails"
    #all_params.append(p)

    file_name = p['expr_name'] + ".log"
    # query the model and save the responses
    #if use_saved_results:
    #    load_results(all_params)
    #else:
    filename = "logging/" + file_name
    f = open(filename, "a")
    save_results(p, output_file=f)
    f.close()

def check_duplicate(arr_a, arr_b):
    for e in arr_a:
        if e in arr_b:
            return True
    return False

def save_results(params, output_file, freeze_test_set=True):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    result_tree = dict()
    all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset_local(params)
    if len(all_train_sentences) > 5000:
        all_train_sentences = all_train_sentences[:5000]
        all_train_labels = all_train_labels[:5000]
    #print(all_train_labels)
    chunk_size = len(all_train_sentences) // 10
    all_valid_sentences = all_train_sentences[chunk_size*9:]
    all_valid_labels = all_train_labels[chunk_size*9:]
    start_index = params["start_index"]
    end_index = params["end_index"]
    all_train_sentences = all_train_sentences[start_index * chunk_size: end_index * chunk_size]
    all_train_labels = all_train_labels[start_index * chunk_size: end_index * chunk_size]
    if params['num_shots'] > 0 and len(all_test_sentences) > chunk_size:
    #all_test_sentences = all_test_sentences[0]
    #all_test_labels = all_test_labels[:20]
        all_test_sentences = all_test_sentences[:chunk_size]
        all_test_labels = all_test_labels[:chunk_size]
    if params['num_shots'] == 0 and len(all_test_sentences) > 300:
        random_index = np.random.choice(len(all_test_labels), 300, replace=False)
        all_test_sentences = [all_test_sentences[i] for i in random_index]
        all_test_labels = [all_test_labels[i] for i in random_index]
    counts = 0
    for sentence in all_test_sentences:
        print(sentence)
        counts += len(sentence.split())
    print(counts / len(all_test_sentences))
    exit()
    #print(all_test_sentences[:10])
    #print(all_valid_labels)
    #exit()
    params_check(params)
    total_prompts = []
    sampled_index = np.random.choice(len(all_train_sentences), params['num_shots']*params['num_trails'], replace=False).tolist()
    print(len(sampled_index))
    for i in range(params['num_trails']):
        print(i)
        #if i == 0:
        #    print("correct train label:", file=output_file, flush = True)
        #    print(all_train_labels, file=output_file, flush = True)
        train_index = sampled_index[i * params['num_shots']: (i+1)* params['num_shots']]
        train_sentences = [all_train_sentences[i] for i in train_index]
        train_labels = [all_train_labels[i] for i in train_index]
        #if len(total_prompts) > 0 or not check_duplicate(
        #train_sentences, train_labels, train_index = random_sampling(all_train_sentences, all_train_labels, params['num_shots'], return_index=True, must_include=params['must_include'])
        #print(train_labels)
        #if params["num_shots"] > 0:
        #    if len(total_prompts) > 0 or not check_duplicate(train_index.tolist(), total_prompts):
        #        total_prompts += train_index.tolist()
        #    else:
        #        continue 
        #raw_resp_test = get_model_response(params, train_sentences, train_labels, all_train_sentences)
        # get prob for each label
        #all_label_probs_train = get_label_probs(params, raw_resp_test, train_sentences, train_labels, all_train_sentences)
        #acc_original_train = eval_accuracy(all_label_probs_train, all_train_labels)
        content_free_inputs = ["N/A", "", "[MASK]"]
        p_cf = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)
        if params["num_shots"] > 0: 
            raw_resp_test = get_model_response(params, train_sentences, train_labels, all_valid_sentences)
        #print(raw_resp_test)
            all_label_probs_test = get_label_probs(params, raw_resp_test, train_sentences, train_labels, all_valid_sentences)
        #acc_original_test = eval_accuracy(all_label_probs_test, all_test_labels)
            #content_free_inputs = ["N/A", "", "[MASK]"]
            #p_cf = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)
            #predicted_labels_valid, acc_calibrated_valid = eval_accuracy(all_label_probs_test, all_valid_labels)
            predicted_labels_valid, acc_calibrated_valid = eval_accuracy(all_label_probs_test, all_valid_labels, mode="diagonal_W", p_cf=p_cf)
        
           
        raw_resp_test = get_model_response(params, train_sentences, train_labels, all_test_sentences)
        #print(raw_resp_test)
        all_label_probs_test = get_label_probs(params, raw_resp_test, train_sentences, train_labels, all_test_sentences)
        #acc_original_test = eval_accuracy(all_label_probs_test, all_test_labels)
        #content_free_inputs = ["N/A", "", "[MASK]"]
        #p_cf = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)
        #predicted_labels_test, acc_calibrated_test = eval_accuracy(all_label_probs_test, all_test_labels)
        predicted_labels_test, acc_calibrated_test = eval_accuracy(all_label_probs_test, all_test_labels, mode="diagonal_W", p_cf=p_cf)
        #print(len(all_label_probs_train), len(all_label_probs_test))
    
        if params['num_shots'] > 0:
            #print("validation accuracy is " + str(acc_original_test), file=output_file, flush = True)
            print("calibrated validation accuracy is " + str(acc_calibrated_valid), file=output_file, flush = True)
            #print("test accuracy is " + str(acc_original_test))
            #print("validation accuracy is " + str(acc_original_test))
            print("calibrated validation accuracy is " + str(acc_calibrated_valid))
            
            print(train_index, file=output_file, flush = True)
            print("labels for the iid public data is ")
            print(predicted_labels_valid, file=output_file, flush = True)
            print("labels for the ood public data is ")
            print(predicted_labels_test, file=output_file, flush = True)
            
        else:
            print("calibrated test accuracy is " + str(acc_calibrated_test))
            #print("here")
            #print(correct_label_probs, file=output_file, flush = True)
            break

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    # print(all_label_probs)
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    #print(len(all_label_probs), len(test_labels))
    assert len(all_label_probs) == len(test_labels)
    predicted_label = [0] * len(test_labels)
    i = 0
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / (np.sum(label_probs)+0.00001) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        predicted_label[i] = ans_label
        i += 1
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    #print(predicted_label)
    #print(true_label)
    return predicted_label, np.mean(correctness_list)


def get_label_probs(params, raw_resp, train_sentences, train_labels, test_sentences):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(params['label_dict'])
    approx = params['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        #print(ans['logprobs']['top_logprobs'])
        if len(ans['logprobs']['top_logprobs']) == 0:
            for j, label_list in params['label_dict'].items():
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
            all_label_probs.append([0] * len(params['label_dict'].keys()))
            continue
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
        label_probs = [0] * len(params['label_dict'].keys())
        for j, label_list in params['label_dict'].items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs) # prob not normalized
    
    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = params['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
                prompt += " " + label
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(params)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):
            resp = complete(chunk, 0, params['model'], echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = np.exp(ans['logprobs']['token_logprobs'][-1])
                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill run_mi_sbatch.shin corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        #assert (all_label_probs > 0).all(), "all should be populated with non-zero value"
    
    return all_label_probs # NOT NORMALIZED

def get_p_content_free(params, train_sentences, train_labels, content_free_inputs=('N/A',)):
    """Query model with content free input, return its prediction probability for each label"""
    label_dict = params['label_dict']

    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt(params, train_sentences, train_labels, content_free_input)

        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1])
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y


def params_check(params):
    """sanity check the experiment params"""
    assert params['num_tokens_to_predict'] == 1
    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            first_token_of_label_name = complete(' ' + label_name, 1, params['model'], echo=True, num_log_probs=2)['choices'][0]['logprobs']['tokens'][0]
            if first_token_of_label_name[1:] != label_name:
                print('label name is more than 1 token', label_name)
                assert False

   # if not (params['dataset'] in ['cb', 'rte']):
        # formatting: there should be a space after question/answer prefix
   #     assert params["q_prefix"][-1] == " "
   #     assert params["a_prefix"][-1] == " "
   #     assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    #parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--train_dataset', dest='train_dataset', action='store', required=True,
                        help='name of dataset(s), e.g., agnews')
    parser.add_argument('--test_dataset', dest='test_dataset', action='store', required=True,
                        help='name of dataset(s), e.g., agnews')

    #parser.add_argument('--transfer_datasets', dest='transfer_datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    parser.add_argument('--must_include', dest='must_include', type=int)
    #parser.add_argument('--obfuscate', dest='obfuscate', type=str, default="None", help='obfuscate')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')

    parser.add_argument('--obfuscate', dest='obfuscate', type=str, default="None", help='obfuscate')

    parser.add_argument('--start_index', dest='start_index', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--end_index', dest='end_index', action='store', required=True, help='num seeds for the training set', type=int)
    args = parser.parse_args()
    args = vars(args)
    np.set_printoptions(threshold=sys.maxsize)
    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    
    args['models'] = convert_to_list(args['models'])
    #args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)

    main(**args)
