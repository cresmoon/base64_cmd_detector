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


USAGE_TEXT = """
Usage: python %s <input_data_csv> <output_model_file>
"""

BATCH_SIZE = 64
HIDDEN_DIM = 64
NUM_EPOCHS = 50


def usage(script_name):
    print(USAGE_TEXT % script_name)
    sys.exit(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, data_loader, optimizer, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_len = 0
    model.train()  # let the model know that it's the training phase
    for batch in data_loader:
        batch_yy, batch_xx, lengths = batch
        if device != 'cpu':
            batch_xx = batch_xx.to(device)
            batch_yy = batch_yy.to(device)
            lengths = lengths.to(device)
        optimizer.zero_grad()
        preds = model(batch_xx, lengths)
        loss = criterion(preds, batch_yy)
        acc, acc_len = esab46.utils.binary_accuracy(preds, batch_yy)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_len += acc_len
    return epoch_loss, epoch_acc / epoch_len


def evaluate_epoch(model, data_loader, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_len = 0
    model.eval()  # let the model know that it's the training phase
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            batch_yy, batch_xx, lengths = batch
            if device != 'cpu':
                batch_xx = batch_xx.to(device)
                batch_yy = batch_yy.to(device)
                lengths = lengths.to(device)
            preds = model(batch_xx, lengths)
            loss = criterion(preds, batch_yy)
            acc, acc_len = esab46.utils.binary_accuracy(preds, batch_yy)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_len += acc_len
    return epoch_loss, epoch_acc / epoch_len


def main(argv):
    script_name = argv[0]
    if len(argv) < 3:
        usage(script_name)

    random_seed = random.randint(1, 10**6)
    print('random_seed:', random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    input_data_file = argv[1]
    given_output_paths = os.path.splitext(argv[2])
    output_data_file = given_output_paths[0] + '_seed_' + str(random_seed) + given_output_paths[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    cmd_token_dataset = esab46.data.CmdLineTokenDataset(input_data_file)
    cmd_token_data_size = len(cmd_token_dataset)
    print('whole data size', cmd_token_data_size)
    train_data_size = int(0.8 * cmd_token_data_size)
    valid_data_size = cmd_token_data_size - train_data_size

    (training_data, validate_data) = torch_data.random_split(cmd_token_dataset,
                                                             [train_data_size, valid_data_size])
    print('train data size:', len(training_data), '-- validate data size:', len(validate_data))

    vocab_size = len(cmd_token_dataset.vocab)
    print('vocab size:', vocab_size)

    pad_idx = cmd_token_dataset.vocab['<pad>']
    collator = esab46.data.CmdLineTokenCollator(pad_idx)
    train_data_loader = torch_data.DataLoader(training_data,
                                              batch_size=BATCH_SIZE,
                                              collate_fn=collator.collate,
                                              shuffle=True,
                                              num_workers=4)
    valid_data_loader = torch_data.DataLoader(validate_data,
                                              batch_size=BATCH_SIZE,
                                              collate_fn=collator.collate,
                                              shuffle=True,
                                              num_workers=4)

    model = esab46.model.JenNet(vocab_size, HIDDEN_DIM)
    print('Number of trainable params of the model:', count_parameters(model))

    criterion = torch.nn.BCEWithLogitsLoss()

    if device != 'cpu':
        model = model.to(device)
        criterion = criterion.to(device)

    # it is recommended that optimizer is constructed after moving model to GPU
    # https://pytorch.org/docs/master/optim.html
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = NUM_EPOCHS
    best_valid_loss = float('inf')
    for ep in range(num_epochs):
        start_time = time.monotonic()
        train_ep_loss, train_ep_acc = train_epoch(model, train_data_loader, optimizer,
                                                  criterion, device)
        valid_ep_loss, valid_ep_acc = evaluate_epoch(model, valid_data_loader, criterion, device)
        end_time = time.monotonic()
        ep_mins, ep_secs = esab46.utils.epoch_time(start_time, end_time)

        print(f'epoch {ep+1:02} {ep_mins}m{ep_secs}s | '
              f'train loss {train_ep_loss:.5f} | train acc {train_ep_acc*100:.3f}% | '
              f'eval loss {valid_ep_loss:.5f} | eval acc {valid_ep_acc*100:.3f}%')

        if valid_ep_loss < best_valid_loss:
            best_valid_loss = valid_ep_loss
            print('saving model to', output_data_file)
            torch.save(model.state_dict(), output_data_file)


if __name__ == '__main__':
    main(sys.argv)
