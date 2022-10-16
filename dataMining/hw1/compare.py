
# %%

fp_file = "/home/q56104076/projects/NCKU-work/dataMining/hw1/ap_result.txt"
ap_file = "/home/q56104076/projects/NCKU-work/dataMining/hw1/fp_result.txt"

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
