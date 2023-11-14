from datasets import load_dataset
import numpy as np
import random
dataset = load_dataset('imdb', split='unsupervised')
orig_train_sentences = dataset["text"]
sample_sentences = random.sample(orig_train_sentences, 5000)
final_sentences = []
for sentence in sample_sentences:
    if "." in sentence:
        sentence = random.sample(sentence.split(".")[:-1], 1)
        if len(sentence[0]) > 5:
            sentence = sentence[0].replace("<br />", "")
            sentence = sentence.strip()
            final_sentences += [sentence]
    #else:
    #    sentence

print(len(final_sentences))
with open(r'data/imdb.txt', 'w') as fp:
    for sentence in final_sentences:
        # write each item on a new line
        fp.write("%s\n" % sentence)
