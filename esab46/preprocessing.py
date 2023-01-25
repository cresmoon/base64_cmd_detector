import base64
import random
from collections import namedtuple

CmdLine = namedtuple('CmdLine', ['exec', 'args'])

FILTERED_LEN_THRESHOLD = 24


def tokenize_args(s):
    return s.split(' ')


def parse_cmd_line(cmd_str):
    cmd_len = len(cmd_str)
    if cmd_str.startswith('"') or cmd_str.startswith('\''):
        quote = cmd_str[0]
        end_of_exe = 1
        for i in range(1, cmd_len):
            if cmd_str[i] == quote:
                end_of_exe = i+1
                assert (end_of_exe == cmd_len) or (cmd_str[end_of_exe] == ' ')
                break
        ret = CmdLine(cmd_str[0:end_of_exe], tokenize_args(cmd_str[end_of_exe+1:]))
    else:
        end_of_exe = cmd_str.find(' ')
        if end_of_exe == -1:
            ret = CmdLine(cmd_str, [])
        else:
            ret = CmdLine(cmd_str[0:end_of_exe], tokenize_args(cmd_str[end_of_exe+1:]))
    return ret


def get_token_len_stats(token_lists):
    total_count = 0
    len_count = {}
    for token_list in token_lists:
        for token in token_list:
            token_len = len(token)
            if token_len not in len_count:
                len_count[token_len] = 0
            len_count[token_len] += 1
            total_count += 1
    return len_count, total_count


def flatten_token_lists(token_lists):
    all_token_list = []
    all_token_count = 0
    for token_list in token_lists:
        all_token_list += token_list
        all_token_count += len(token_list)
    assert all_token_count == len(all_token_list)
    return all_token_list


def is_eligible(token):
    if len(token) < FILTERED_LEN_THRESHOLD:
        return False
    # check if the whole token is non-alphanumeric
    non_alnum = True
    for c in token:
        if c.isalnum():
            non_alnum = False
    if non_alnum:
        # the whole token is non-alphanumeric
        print('Ignoring non-alphanumeric token:', token)
        return False
    return True


def filter_tokens(token_list):
    filtered_token_list = []
    for token in token_list:
        if not is_eligible(token):
            continue
        filtered_token_list.append(token)
    return filtered_token_list


def shuffle_token(token):
    char_list = list(token)
    random.shuffle(char_list)
    return ''.join(char_list)


def convert_token_to_b64(token):
    token = shuffle_token(token)
    return base64.b64encode(token.encode('ascii')).decode()


def get_base64_tokens(token_list):
    base64_list = []
    for token in token_list:
        b64 = convert_token_to_b64(token)
        base64_list.append(b64)
    return base64_list
