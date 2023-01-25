import sys

USAGE_TEXT = """
Usage: python %s <input_data_file>
"""


def usage(script_name):
    print(USAGE_TEXT % script_name)
    sys.exit(-1)


def main(argv):
    script_name = argv[0]
    if len(argv) < 2:
        usage(script_name)
    input_data_file = argv[1]
    data_len = []
    with open(input_data_file, 'rb') as in_file:
        for line in in_file:
            data_len.append(len(line.strip()))
    print('Num. tokens: ', len(data_len))
    print('Max token len: ', max(data_len))


if __name__ == '__main__':
    main(sys.argv)
