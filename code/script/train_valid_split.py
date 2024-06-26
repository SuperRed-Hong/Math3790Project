import argparse
import os
import re

import MCCWS.preprocess
import MCCWS.util


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split_dataset",
        default="../data/trainset/pku_train.txt",
        help="Give the path of dataset file which you want to split.",
        type=str,
    )
    parser.add_argument(
        "--new_train_file_path",
        default="../data/trainset/pku_train.txt",
        help="Give the path where to store the splited train set.",
        type=str,
    )
    parser.add_argument(
        "--new_valid_file_path",
        default="../data/trainset/pku_valid.txt",
        help="Give the path where to store the splited valid set.",
        type=str,
    )

    return parser.parse_args()


def main(args):
    MCCWS.util.set_seed(42)
    dataset = {"[SET]": [args.split_dataset]}
    assert (
        args.split_dataset != ""
        and args.new_train_file_path != ""
        and args.new_train_file_path != ""
    )
    preprocessed_dataset = MCCWS.preprocess.split_dataset(datasets=dataset)

    dir_name = re.findall(r"([^/]+)/[^/]+$", args.new_train_file_path)
    try:
        os.makedirs(f"{dir_name[0]}")
    except:
        pass

    with open(args.split_dataset, "w") as f:
        for data in preprocessed_dataset.data:
            f.write(data + "\n")
    with open(args.new_train_file_path, "w") as f:
        for data in preprocessed_dataset.train:
            f.write(data + "\n")
    with open(args.new_valid_file_path, "w") as f:
        for data in preprocessed_dataset.valid:
            f.write(data + "\n")


if __name__ == "__main__":
    main(args=get_args())
