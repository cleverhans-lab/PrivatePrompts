import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', dest='file_name', action='store', required=True)
parser.add_argument('--label_file', dest='label_file', action='store', required=True)
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
label_file = open(args["label_file"], "r")
labels = [int(a) for a in label_file.readlines()[0][1:-2].split(",")]
count = 0
start = False
start_iid = False
start_ood = False
# Strips the newline character
for line in Lines:
    if line.startswith("calibrated validation accuracy"):
        validation_accuracy += [float(line[len("calibrated validation accuracy is "):])]
        start = True
    elif start:
        start_iid = True
        start = False
    elif start_iid:
        line = line[1:-2].strip()
        votes_iid.append([int(a) for a in line.split(",")])
        start_iid = False
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

# Find index of maximum count
    max_count_index = np.argmax(counts)

# Retrieve the most frequent item
    most_frequent_item = unique_elements[max_count_index]
    if most_frequent_item == labels[i]:
        accuracy += 1
print(accuracy / 500)

validation_accuracy = np.asarray(validation_accuracy)
#print(np.mean(validation_accuracy))

if not args['topk'] is None:
    keep_index = np.argsort(validation_accuracy)[-int(args['topk']):]
    validation_accuracy = [validation_accuracy[i] for i in keep_index]
    votes_iid = [votes_iid[i] for i in keep_index]
    votes_ood = [votes_ood[i] for i in keep_index]
print(len(validation_accuracy), np.max(validation_accuracy), np.std(validation_accuracy))

#np.savetxt("prompt_index/trec_whole.txt", prompt_index, fmt="%i")
np.savetxt(args["output_name_iid"], votes_iid, fmt="%i")
np.savetxt(args["output_name_ood"], votes_ood, fmt="%i")

