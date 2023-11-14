from datasets import load_dataset
import numpy as np
import random
dataset = load_dataset('okite97/news-data', split='train')
orig_train_sentences = dataset["Excerpt"]
print(orig_train_sentences)
random_index = np.random.choice(len(orig_train_sentences), 1000, replace=False)
sample_sentences = [orig_train_sentences[i] for i in random_index]
#sample_sentences = random.sample(orig_train_sentences, 1000)
#final_sentences = []
"""
for sentence in sample_sentences:
    if "." in sentence:
        sentence = random.sample(sentence.split(".")[:-1], 1)
        if len(sentence[0]) > 5:
            sentence = sentence[0].replace("<br />", "")
            sentence = sentence.strip()
            final_sentences += [sentence]
    #else:
    #    sentence
"""
print(len(final_sentences))
with open(r'data/news.txt', 'w') as fp:
    for sentence in sample_sentences:
        # write each item on a new line
        fp.write("%s\n" % sentence)
