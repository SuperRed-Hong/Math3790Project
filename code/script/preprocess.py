import argparse
import os
import re

import MCCWS.preprocess
import MCCWS.util


# ../../icwb2-data/gold/pku_training_words.utf8
# ../data/trainset/pku_train.txt
#
# ../../icwb2-data/gold/pku_test_gold.utf8
# ../data/testset/pku_test.txt

# "../../icwb2-data/testing/pku_test.utf8"
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_gold_path",
        default="../../icwb2-data/gold/pku_test_gold.utf8",
        help="Give the path of test gold file which you want to preprocess.",
        type=str,
    )
    parser.add_argument(
        "--new_path",
        default="../data/testset/pku_test.txt",
        help="Give the path where to store the preprocessed test file.",
        type=str,
    )

    return parser.parse_args()


def main(args):
    MCCWS.util.set_seed(42)

    dir_name = re.findall(r"(.+)/[^/]+$", args.new_path)
    try:
        os.makedirs(f"{dir_name[0]}")
    except:
        pass

    dataset = {"[SET]": [args.original_gold_path]}
    preprocessed_dataset = MCCWS.preprocess.preprocess(datasets=dataset)

    with open(f"{args.original_gold_path}", "w") as f:
        for data in preprocessed_dataset.original_data:
            f.write(data + "\n")
    with open(args.new_path, "w") as f:
        for data in preprocessed_dataset.data:
            f.write(data + "\n")


if __name__ == "__main__":
    main(args=get_args())
