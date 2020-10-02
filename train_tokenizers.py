import sys
import os
import json
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

def train_tokenizer(filename, params):
    """
    Train a BertWordPieceTokenizer with the specified params and save it
    """
    # Get tokenization params
    save_location = params["tokenizer_path"]
    max_length = params["max_length"]
    min_freq = params["min_freq"]
    vocabsize = params["vocab_size"]

    tokenizer = BertWordPieceTokenizer()
    tokenizer.do_lower_case = False
    special_tokens = ["[S]","[PAD]","[/S]","[UNK]","[MASK]", "[SEP]","[CLS]"]
    tokenizer.train(files=[filename], vocab_size=vocabsize, min_frequency=min_freq, special_tokens = special_tokens)

    tokenizer._tokenizer.post_processor = BertProcessing(("[SEP]", tokenizer.token_to_id("[SEP]")), ("[CLS]", tokenizer.token_to_id("[CLS]")),)
    tokenizer.enable_truncation(max_length=max_length)

    print("Saving tokenizer ...")
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    tokenizer.save(save_location)

# Identify the config to use
if len(sys.argv) < 2:
    print("No config file specified. Using the default config.")
    configfile = "config.json"
else:
    configfile = sys.argv[1]

# Read the params
with open(configfile, "r") as f:
    config = json.load(f)

globalparams = config["global_params"]
encparams = config["encoder_params"]
decparams = config["decoder_params"]

# Get the dataset files
train_en_file = globalparams["train_en_file"]
train_de_file = globalparams["train_de_file"]

# Train the tokenizers
train_tokenizer(train_en_file, encparams)
train_tokenizer(train_de_file, decparams)
