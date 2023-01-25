import sys
import esab46.preprocessing

USAGE_TEXT = """
Usage: python %s <input_data_file>
"""


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
    script_name = argv[0]
    if len(argv) < 2:
        usage(script_name)
    input_data_file = argv[1]
    cmd_count, cmd_list, exe_list, token_list = get_arg_token_data(input_data_file)
    print('Number of cmd strings:', cmd_count)
    token_len_counter, token_total_count = esab46.preprocessing.get_token_len_stats(token_list)
    cumulative_count = 0
    print('token len\tcount\tfraction\tcumulative')
    for k in sorted(token_len_counter.keys()):
        cumulative_count += token_len_counter[k]
        print('{0}\t{1}\t{2}\t{3}'.format(k, token_len_counter[k],
                                          token_len_counter[k]/token_total_count,
                                          cumulative_count/token_total_count))


if __name__ == '__main__':
    main(sys.argv)
