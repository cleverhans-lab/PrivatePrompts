# PromptDPSGD

To install the required packages navigate to `install.txt`.

An example of how to run the code is in `run.sh` file.

The code in private_transformers directory comes from: https://github.com/lxuechen/private-transformers and is in this submission only to enable the run of our code.

This code in dp_transformers comes from https://github.com/microsoft/dp-transformers and is in this submission only to enable the run of our code.

The main privacy engine in this work is from:  `private_transformers.privacy_engine.PrivacyEngine`.

The models used are in the `model.sequence_classification` module and the specific classes used in our experiments in the paper are:

- RobertaPrefixForSequenceClassification
- RobertaPromptForSequenceClassification




