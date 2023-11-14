import numpy as np
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', dest='file_name', action='store', required=True)
args = parser.parse_args()
args = vars(args)
validation_accuracy = []
test_accuracy = []

file = open(args["file_name"], 'r')
Lines = file.readlines()

count = 0
# Strips the newline character
for line in Lines:
    if line.startswith("validation accuracy"):
        validation_accuracy += [float(line[len("validation accuracy is "):])]
    elif line.startswith("test"):
        test_accuracy += [float(line[len("test accuracy is "):])]


validation_accuracy = np.asarray(validation_accuracy)
test_accuracy = np.asarray(test_accuracy)
df = pd.DataFrame({'validation accuracy': validation_accuracy, 'test accuracy': test_accuracy})
df.to_csv('student_sst2_imdb.csv', index=False)
#print(validation_accuracy)
print(len(test_accuracy))
best_valid = np.argsort(validation_accuracy)[-1]
print(test_accuracy[best_valid])
#np.savetxt(args["output_name"], selected_prompt, fmt="%i", delimiter=" ")
