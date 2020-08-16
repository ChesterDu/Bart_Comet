import os
import sys
import argparse
import demo_bilinear
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--gens_name", type=str, default="/content/drive/My Drive/bart_finetune/ckbc-demo/results.txt")
parser.add_argument("--thresh", type=float, default=0.5)

args = parser.parse_args()

# print(gens_file[0])
results = demo_bilinear.run(args.gens_name, flip_r_e1=True)
new_results = {"0": [j for (i, j) in results if i[3] == "0"],
               "1": [j for (i, j) in results if i[3] == "1"]}
print(new_results)
# print("Total")
num_examples = 1.0 * len(results)
# print(num_examples)
positive = sum(np.array(new_results["1"]) > args.thresh)
# accuracy = (len([i for i in new_results["1"] if i >= args.thresh]) +
#             len([i for i in new_results["0"] if i < args.thresh])) / num_examples
accuracy = positive / num_examples
print("Accuracy @ {}: {}".format(args.thresh, accuracy))

