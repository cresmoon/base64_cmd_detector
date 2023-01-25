import sys
import os

USAGE_TEXT = """
Usage: python %s <input_data_file> <output_data_file>
"""


def usage(script_name):
    print(USAGE_TEXT % script_name)
    sys.exit(-1)


def get_hash_to_data_map(input_file_path):
    hash_to_cmd = {}
    with open(input_file_path) as input_file:
        for line in input_file:
            cmd = line.strip()
            h = hash(cmd)
            if h in hash_to_cmd:
                if len(hash_to_cmd[h]) != len(cmd):
                    print('Collision: {} vs. {}'.format(hash_to_cmd[h], cmd))
            else:
                hash_to_cmd[h] = cmd
    return hash_to_cmd


def write_hash_to_data_map(hash_to_cmd, output_file_path):
    with open(output_file_path, 'w') as output_file:
        for cmd in hash_to_cmd.values():
            output_file.write(cmd + '\n')


def main(argv):
    script_name = argv[0]
    if len(argv) < 3:
        usage(script_name)
    input_data_file = argv[1]
    output_data_file = argv[2]
    hash_to_data = get_hash_to_data_map(input_data_file)
    write_hash_to_data_map(hash_to_data, output_data_file)


if __name__ == '__main__':
    main(sys.argv)
