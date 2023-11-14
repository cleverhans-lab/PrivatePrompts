import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', dest='file_name', action='store', required=True)
parser.add_argument('--train_index', dest='train_index', action='store')
parser.add_argument('--output_name', dest='output_name', action='store', required=True)
parser.add_argument('--output_index', dest='output_index', action='store')
args = parser.parse_args()
args = vars(args)
public_votes = []
eliminated_index = None
if not args["train_index"] is None:
    eliminated_index = np.loadtxt(args["train_index"], dtype=int)

    if eliminated_index.ndim > 1:
        eliminated_index = np.unique(eliminated_index.flatten())
file = open(args["file_name"], 'r')
Lines = file.readlines()
selected_index = None
count = 0
# Strips the newline character
read_labels = False
for line in Lines:
    if line.startswith("labels for the test set"):
        read_labels = True
    elif read_labels:
        label = [int(index) for index in line[1:-2].split(",")]
        #print(len(label))
        if not eliminated_index is None:
            selected_index = np.setdiff1d(np.arange(len(label)), eliminated_index)
            label = np.asarray(label)[selected_index].tolist()
        print(len(label))
        public_votes.append(label)
        read_labels = False

if not selected_index is None:
    np.savetxt(args["output_index"], selected_index, fmt="%i", delimiter=" ")
np.savetxt(args["output_name"], public_votes, fmt="%i", delimiter=" ")
