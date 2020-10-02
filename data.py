import sys
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import os

class TranslationDataset(data.Dataset):

    def __init__(self, inp_file, targ_file, inp_tokenizer, targ_tokenizer, inp_maxlength, targ_maxlength):

        self.inp_tokenizer = inp_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.inp_maxlength = inp_maxlength
        self.targ_maxlength = targ_maxlength

        print("Loading and Tokenizing the data ...")
        self.encoded_inp = []
        self.encoded_targ = []

        # Read the EN lines
        num_inp_lines = 0
        with open(inp_file, "r") as ef:
            for line in ef:
                enc = self.inp_tokenizer.encode(line.strip(), add_special_tokens=True, max_length=self.inp_maxlength)
                self.encoded_inp.append(torch.tensor(enc))
                num_inp_lines += 1

        # read the DE lines
        num_targ_lines = 0
        with open(targ_file, "r") as df:
            for line in df:
                enc = self.targ_tokenizer.encode(line.strip(), add_special_tokens=True, max_length=self.targ_maxlength)
                self.encoded_targ.append(torch.tensor(enc))
                num_targ_lines += 1

        assert (num_inp_lines==num_targ_lines), "Mismatch in EN and DE lines"
        print("Read", num_inp_lines, "lines from EN and DE files.")

    def __getitem__(self, offset):
        en = self.encoded_inp[offset]
        de = self.encoded_targ[offset]

        return en, en.shape[0], de, de.shape[0]

    def __len__(self):
        return len(self.encoded_inp)

    def collate_function(self, batch):

        (inputs, inp_lengths, targets, targ_lengths) = zip(*batch)

        padded_inputs = self._collate_helper(inputs, self.inp_tokenizer)
        padded_targets = self._collate_helper(targets, self.targ_tokenizer)

        max_inp_seq_len = padded_inputs.shape[1]
        max_out_seq_len = padded_targets.shape[1]

        input_masks = [[1]*l + [0]*(max_inp_seq_len-l) for l in inp_lengths]
        target_masks = [[1]*l + [0]*(max_out_seq_len-l) for l in targ_lengths]

        input_tensor = padded_inputs.to(torch.int64)
        target_tensor = padded_targets.to(torch.int64)
        input_masks = torch.Tensor(input_masks)
        target_masks = torch.Tensor(target_masks)

        return input_tensor, input_masks, target_tensor, target_masks

    def _collate_helper(self, examples, tokenizer):
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
