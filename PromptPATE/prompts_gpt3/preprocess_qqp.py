from datasets import load_dataset
import numpy as np
import random
dataset = load_dataset('glue','qqp', split='train')
orig_train_sentences = dataset["question2"]
sample_sentences = None
if len(orig_train_sentences) > 6000:
    sample_sentences = random.sample(orig_train_sentences, 6000)
else:
    sample_sentences = orig_train_sentences
final_sentences = []
for sentence in sample_sentences:
    #print(sentence)
    if not (". " in sentence or "," in sentence):
        final_sentences += [sentence]
    #print(final_sentences[-1])
print(final_sentences[:10])
#print(len(final_sentences))
with open(r'data/qqp.txt', 'w') as fp:
    for sentence in final_sentences:
        # write each item on a new line
        fp.write("%s\n" % sentence)
