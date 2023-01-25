import sys

USAGE_TEXT = """
Usage: python %s <input_data_file>
"""


def usage(script_name):
    print(USAGE_TEXT % script_name)
    sys.exit(-1)


def get_special_char_count(input_file_path):
    cmd_count = 0
    special_char_count = {}

    with open(input_file_path) as input_file:
        for line in input_file:
            cmd = line.strip()
            for c in cmd:
                if not c.isalnum():
                    if c not in special_char_count:
                        special_char_count[c] = 0
                    special_char_count[c] += 1
            cmd_count += 1
    
    return cmd_count, special_char_count


def main(argv):
    script_name = argv[0]
    if len(argv) < 2:
        usage(script_name)
    input_data_file = argv[1]
    cmd_count, special_char_count = get_special_char_count(input_data_file)
    print('Number of cmd strings:', cmd_count)
    for k, v in special_char_count.items():
        print('Special char {0} count: {1}'.format(k, v))
    print(''.join(special_char_count.keys()))


if __name__ == '__main__':
    main(sys.argv)
