import os
import sys
import random
import time
import numpy as np
import torch
from torch.utils import data as torch_data
import esab46.model
import esab46.data
import esab46.utils
import esab46.preprocessing


USAGE_TEXT = """
Usage: python %s <input_data_csv> <input_model_file>
"""

BATCH_SIZE = 64
HIDDEN_DIM = 64


def usage(script_name):
    print(USAGE_TEXT % script_name)
    sys.exit(-1)


def main(argv):
    script_name = argv[0]
    if len(argv) < 3:
        usage(script_name)

    input_data_file = argv[1]
    input_model_file = argv[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    cmd_token_dataset = esab46.data.CmdLineTokenDataset(input_data_file)
    cmd_token_data_size = len(cmd_token_dataset)
    print('whole data size', cmd_token_data_size)
    vocab_size = len(cmd_token_dataset.vocab)
    print('vocab size:', vocab_size)

    pad_idx = cmd_token_dataset.vocab['<pad>']
    collator = esab46.data.CmdLineTokenCollator(pad_idx)
    data_loader = torch_data.DataLoader(cmd_token_dataset,
                                        batch_size=BATCH_SIZE,
                                        collate_fn=collator.collate,
                                        shuffle=True,
                                        num_workers=4)
    # loading model
    model = esab46.model.JenNet(vocab_size, HIDDEN_DIM)
    model.load_state_dict(torch.load(input_model_file, map_location=device))
    if device != 'cpu':
        model = model.to(device)
    model.eval()  # set the eval mode

    total_acc = 0
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            batch_yy, batch_xx, lengths = batch
            if device != 'cpu':
                batch_xx = batch_xx.to(device)
                batch_yy = batch_yy.to(device)
                lengths = lengths.to(device)
            preds = model(batch_xx, lengths)
            acc, _ = esab46.utils.binary_accuracy(preds, batch_yy)
            total_acc += acc.item()

    print(f'testing size {cmd_token_data_size} | '
          f'testing correct {int(total_acc)} | '
          f'testing acc {total_acc/cmd_token_data_size * 100:.4f}%')


if __name__ == '__main__':
    main(sys.argv)
