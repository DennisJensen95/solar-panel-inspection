import os
import re
import argparse as ap


def replace_model(model, file_content):
    file_content.replace("MODEL_PLACEHOLDER", model)


def replace_labels(labels, file_content):
    file_content.replace("LABELS_PLACEHOLDER", labels)


def configure_training_model(model, labels):
    with open("train.py", "r") as file:
        file_contents = file.read()

    file_contents = replace_model(model, file_contents)
    file_contents = replace_labels(labels, file_contents)


def setup_arg_parsing():
    parser = ap.ArgumentParser(description="Training script")
    parser.add_argument("--ModelConf")
    args, leftovers = parser.parse_known_args()

    return args
