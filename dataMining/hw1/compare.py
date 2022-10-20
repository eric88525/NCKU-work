
# %%

import os
from genericpath import isfile


def compare(fp_file, ap_file):

    print("Compare ...", ap_file, fp_file)

    if not (os.path.isfile(fp_file)):
        print("File not found:", fp_file)
        return

    if not (os.path.isfile(ap_file)):
        print("File not found:", ap_file)
        return

    fp_dict = set()

    diff = []

    with open(fp_file, "r") as f:
        fp_rule = f.readlines()
        for line in fp_rule:

            fp_dict.add(line.replace("\n", ""))

    with open(ap_file, "r") as f:
        ap_rule = f.readlines()

        for line in ap_rule:
            line = line.replace("\n", "")
            if line not in fp_dict:
                diff.append(line)

    if len(diff) == 0:
        print("No difference")
    else:
        for line in diff:
            print(line)

# %%


if __name__ == "__main__":

    fp_file = "./ap_result.txt"
    ap_file = "./fp_result.txt"
    compare(fp_file, ap_file)

    fp_file = "./ap_rule.txt"
    ap_file = "./fp_rule.txt"
    compare(fp_file, ap_file)
