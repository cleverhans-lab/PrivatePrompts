import numpy as np
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', dest='file_name', action='store', required=True)
parser.add_argument('--dataset', dest='dataset', action='store', required=True)
parser.add_argument('--topk', dest='topk', action='store')
parser.add_argument('--output_name_iid', dest='output_name_iid', action='store', required=True)
parser.add_argument('--output_name_ood', dest='output_name_ood', action='store', required=True)
args = parser.parse_args()
args = vars(args)
validation_accuracy = []
votes_iid = []
votes_ood = []
prompt_index = []
file = open(args["file_name"], 'r')
Lines = file.readlines()
labels = None
if args['dataset'] == 'sst2':
    datasets = load_dataset("sst2", split="train")
    labels = datasets["label"][:200]
if args['dataset'] == 'agnews':
    datasets = load_dataset("ag_news", split="test")
    labels = datasets["label"][:200]
if args['dataset'] == 'trec':
    datasets = load_dataset("trec", split="train")
    labels = datasets["coarse_label"][:200]

count = 0
start_valid = False
start_iid = False
start_ood = False
# Strips the newline character
for line in Lines:
    if line.startswith("validation accuracy"):
        start_valid = True
    elif start_valid:
        validation_accuracy.append(float(line[:-1]))
        start_valid = False
    elif line.startswith("labels for the iid"):
        start_iid = True
    elif start_iid:
        line = line[1:-2].strip()
        votes_iid.append([int(a) for a in line.split(",")])
        start_iid = False
    elif line.startswith("labels for the ood"):
        start_ood = True
    elif start_ood:
        line = line[1:-2].strip()
        votes_ood.append([int(a) for a in line.split(",")])
        start_ood = False

accuracy = 0
votes_iid_np = np.asarray(votes_iid)
for i in range(len(votes_iid[0])):
    votes = votes_iid_np[:, i]
    unique_elements, counts = np.unique(votes, return_counts=True)
    max_count_index = np.argmax(counts)
    most_frequent_item = unique_elements[max_count_index]
    if most_frequent_item == labels[i]:
        accuracy += 1
votes_iid = np.asarray(votes_iid)
votes_ood = np.asarray(votes_ood)
def replace_elements(array, a, b):
    mask = array == a
    array[mask] = b
    return array
max_class = np.max(votes_iid)+1
votes_iid = replace_elements(votes_iid, -2, max_class)
votes_ood = replace_elements(votes_ood, -2, max_class)

print("ensemble accuracy")
#print(votes_iid, labels)
print(accuracy / 200)

validation_accuracy = np.asarray(validation_accuracy)
print(np.sort(validation_accuracy))
print(len(validation_accuracy), np.max(validation_accuracy), np.std(validation_accuracy))

#np.savetxt("prompt_index/trec_whole.txt", prompt_index, fmt="%i")
np.savetxt(args["output_name_iid"], votes_iid, fmt="%i")
#np.savetxt(args["output_name_ood"], votes_ood, fmt="%i")

