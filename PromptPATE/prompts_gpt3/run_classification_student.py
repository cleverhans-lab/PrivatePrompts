import argparse
from data_utils import load_dataset_local
from utils import *
import torch.nn.functional as F
from scipy.special import softmax
import random
import numpy as np
#numpy.set_printoptions(threshold=sys.maxsize)

def main(models, train_dataset, test_dataset, all_shots, num_seeds, subsample_test_set, api_num_log_prob, approx, use_saved_results, bs, obfuscate, must_include, index_file, label_file):
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
    #p['expr_name'] = f"student_apr_27_{random.randint(0, 100000)}_eval_probs_{p['train_dataset']}_{p['test_dataset']}_{p['model']}_{p['num_shots']}_shot_{num_seeds}_{p['obfuscate']}_include_{p['must_include']}_trails"
    #all_params.append(p)

    #file_name = p['expr_name'] + ".log"
    # query the model and save the responses
    #if use_saved_results:
    #    load_results(all_params)
    #else:
    identical = False
    if train_dataset == test_dataset:
        identical = True
    #train_indices = np.loadtxt(index_file, dtype=int)
    train_labels = np.loadtxt(label_file, dtype=int)
    if index_file != "None":
        #print("here")
        #exit()
        train_indices = np.loadtxt(index_file, dtype=int)
    else:
        train_indices = np.arange(len(train_labels))
    p['expr_name'] = f"student_may_10_{random.randint(0, 100000)}_eval_probs_{p['train_dataset']}_{p['test_dataset']}_{p['model']}_{p['num_shots']}_shot_{num_seeds}_{len(train_indices)}_queries"
    file_name = p['expr_name'] + ".log"
    filename = "logging/" + file_name
    f = open(filename, "a")
    save_results(p, output_file=f, train_indices = train_indices, train_labels=train_labels, identical = identical)
    #p['expr_name'] = f"student_apr_27_{random.randint(0, 100000)}_eval_probs_{p['train_dataset']}_{p['test_dataset']}_{p['model']}_{p['num_shots']}_shot_{num_seeds}_{len(train_indices)}_queries"
    f.close()


def save_results(params, output_file, train_indices, train_labels, identical, freeze_test_set=True):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    result_tree = dict()
    all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset_local(params)
    params_check(params)
    num_indices = len(train_indices)
    if identical and len(all_train_sentences) > 5000:
        all_train_sentences = all_train_sentences[:5000]
        all_train_labels = all_train_labels[:5000]
    #print(train_indices)
    if identical:
        #print("here")
        chunk_size = len(all_train_sentences) // 10
        all_train_sentences = all_train_sentences[chunk_size*9:]
        #all_valid_labels = all_train_labels[chunk_size*8:]
    all_train_sentences = [all_train_sentences[i] for i in train_indices]
    #print(all_train_sentences[:10])
        #all_train_labels = [all_train_labels[i]for i in train_indices]
    #else:
        #print(here)
        #all_train_sentences = [all_train_sentences[i+500] for i in train_indices]
        #all_train_labels = [all_train_labels[i+500] for i in train_labels]
    if len(all_test_labels) > 300:
        random_index = np.random.choice(len(all_test_labels), 300, replace=False)
        all_test_sentences = [all_test_sentences[i] for i in random_index]
        all_test_labels = [all_test_labels[i] for i in random_index]
    all_train_labels = train_labels
    #print(all_test_labels)
    #`print(all_train_labels[:50], all_train_sentences[:50])
    assert(len(all_train_labels) == len(all_train_sentences))
    valid_sentences = all_train_sentences
    valid_labels = all_train_labels
    #print(valid_labels)
    if len(all_train_sentences) > 300:
        random_index = np.random.choice(len(valid_labels), 300, replace=False)
        valid_sentences = [valid_sentences[i] for i in random_index]
        valid_labels = [valid_labels[i] for i in random_index]
    #print(_sentences[:10])
    #print(valid_labels[:10])
    #exit()
    best_validation = 0
    best_prompt = None
    acc_original_train = 0
    acc_calibrated = 0
    best_prompts = None
    best_prompts_labels = None
    #print(all_test_sentences[:10])
    if train_indices.ndim == 1:
        train_indices = train_indices.reshape((-1, 1))
    for i in range(params['num_trails']):
        #print(i)
        train_sentences, train_labels, train_index = random_sampling(all_train_sentences, all_train_labels, params['num_shots'], return_index=True, must_include=params['must_include'])
        #print(train_sentences, train_labels)
        raw_resp_test = get_model_response(params, train_sentences, train_labels, valid_sentences)
        # get prob for each label
        all_label_probs_train = get_label_probs(params, raw_resp_test, train_sentences, train_labels,valid_sentences)
        content_free_inputs = ["N/A", "", "[MASK]"]
        p_cf = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)
        #if params["test_dataset"] != "sst2":
        l, acc_original_train = eval_accuracy(all_label_probs_train, valid_labels, mode="diagonal_W", p_cf=p_cf)
        #print(l)
        #print(valid_labels)
        #exit()
        if acc_original_train > best_validation:
            best_validation = acc_original_train
            best_prompts = train_sentences
            best_prompts_labels = train_labels
        if i == params['num_trails']-1:
        #else:
        #    _, acc_original_train = eval_accuracy(all_label_probs_train, valid_labels)
            raw_resp_test = get_model_response(params, best_prompts, best_prompts_labels, all_test_sentences)
            all_label_probs_test = get_label_probs(params, raw_resp_test, best_prompts, best_prompts_labels, all_test_sentences)
        #content_free_inputs = ["N/A", "", "[MASK]"]
        #p_cf = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)
        #if params["test_dataset"] != "sst2":
            content_free_inputs = ["N/A", "", "[MASK]"]
            p_cf = get_p_content_free(params, best_prompts, best_prompts_labels, content_free_inputs=content_free_inputs)
            predicted_labels, acc_calibrated = eval_accuracy(all_label_probs_test, all_test_labels, mode="diagonal_W", p_cf=p_cf)
        #else:
        #    predicted_labels, acc_calibrated = eval_accuracy(all_label_probs_test, all_test_labels)
        #acc_original_test = eval_accuracy(all_label_probs_test, all_test_labels)
        #print(len(all_label_probs_train), len(all_label_probs_test))
        if params['num_shots'] > 0:
            print("validation accuracy is " + str(acc_original_train), file=output_file, flush=True)
            print("validation accuracy is " + str(acc_original_train))

            if i == params['num_trails']-1:
                #print("test ")
                #print(best_prompts, best_prompts_labels)
                print("best validation accuracy " + str(best_validation))
                print("best validation accuracy " + str(best_validation), file=output_file, flush=True)
                print("test accuracy is " + str(acc_calibrated), file=output_file, flush=True)
                print("test accuracy is " + str(acc_calibrated))                                       #print(train_index, file=output_file, flush=True)
            # train_index = train_index.astype(int)
            #print("labels for the test set is:", file=output_file, flush=True)
            # print(all_label_probs_train)

            #print(predicted_labels, file=output_file, flush=True)
            # print("probs for the test set is:", file=output_file)
            # print(all_label_probs_test, file=output_file)
            print("-----------------", file=output_file)
        else:
            # print("here")
            print(correct_label_probs, file=output_file)
            break

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
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

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"
    #print(len(all_label_probs[0]))
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
    """
    if not (params['dataset'] in ['cb', 'rte']):
        # formatting: there should be a space after question/answer prefix
        assert params["q_prefix"][-1] == " "
        assert params["a_prefix"][-1] == " "
        assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'
    """
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    parser.add_argument('--train_dataset', dest='train_dataset', action='store', required=True,
                        help='name of dataset(s), e.g., agnews')
    parser.add_argument('--test_dataset', dest='test_dataset', action='store', required=True,
                        help='name of dataset(s), e.g., agnews')
    parser.add_argument("--index_file", dest="index_file", action = "store", required = True, type=str)
    parser.add_argument("--label_file", dest="label_file", action="store", required=True, type=str)
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
    #args['train_datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)

    main(**args)
