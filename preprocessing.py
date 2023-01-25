import os
import random
import sys
import esab46.preprocessing

USAGE_TEXT = """
Usage: python %s <input_data_txt> <output_data_4_training_csv>
"""

NUM_RANDOM_SHOW = 10


def usage(script_name):
    print(USAGE_TEXT % script_name)
    sys.exit(-1)


def get_arg_token_data(input_file_path):
    cmd_count = 0
    cmd_list = []
    exe_list = []
    arg_token_list = []
    with open(input_file_path) as input_file:
        for line in input_file:
            cmd = line.strip()
            ret_cmd_line = esab46.preprocessing.parse_cmd_line(cmd)
            cmd_list.append(cmd)
            exe_list.append(ret_cmd_line.exec)
            arg_token_list.append(ret_cmd_line.args)
            cmd_count += 1
    return cmd_count, cmd_list, exe_list, arg_token_list


def main(argv):
    random_seed = random.randint(1, 10**6)
    random.seed(random_seed)
    script_name = argv[0]
    if len(argv) < 3:
        usage(script_name)
    input_data_file = argv[1]
    given_output_paths = os.path.splitext(argv[2])
    output_data_file = given_output_paths[0] + '_seed_' + str(random_seed) + given_output_paths[1]
    cmd_count, cmd_list, exe_list, token_lists = get_arg_token_data(input_data_file)
    print('Number of cmd strings:', cmd_count)
    for _ in range(NUM_RANDOM_SHOW):
        n = random.randrange(cmd_count)
        print(cmd_list[n], '==>', exe_list[n], token_lists[n])

    all_token_list = esab46.preprocessing.flatten_token_lists(token_lists)
    all_token_count = len(all_token_list)
    print('Number of tokens:', all_token_count)
    for _ in range(NUM_RANDOM_SHOW):
        n = random.randrange(all_token_count)
        print('A random token:', all_token_list[n])

    filtered_token_list = esab46.preprocessing.filter_tokens(all_token_list)
    filtered_token_count = len(filtered_token_list)
    base64_token_list = esab46.preprocessing.get_base64_tokens(filtered_token_list)
    base64_token_count = len(base64_token_list)
    print('Number of filtered tokens:', filtered_token_count)
    print('Number of base64 tokens:', base64_token_count)
    for _ in range(NUM_RANDOM_SHOW):
        n = random.randrange(filtered_token_count)
        print('A random token:', filtered_token_list[n])
        print('And its base64:', base64_token_list[n])

    assert filtered_token_count == base64_token_count
    print('Write training data to', output_data_file)
    with open(output_data_file, 'w') as out_file:
        for i in range(filtered_token_count):
            out_file.write('0,{}\n'.format(filtered_token_list[i]))
            out_file.write('1,{}\n'.format(base64_token_list[i]))


if __name__ == '__main__':
    main(sys.argv)
