# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

# do not print INFO and WARNING messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow.compat.v1 as tf_v1
except ImportError:
    import tensorflow as tf_v1

#in some environment suppressing through TF_CPP_MIN_LOG_LEVEL does not work
tf_v1.get_logger().setLevel("ERROR")

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from openvino.tools.mo.front.tf.loader import load_tf_graph_def

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to_pbtxt", dest='pb', type=str, help="Path to TensorFlow binary model")
    parser.add_argument('--to_pb', dest='pbtxt', type=str, help="Path to TensorFlow text model")
    return parser.parse_args()


def convert(filename: str, is_text: bool):
    if not os.path.isfile(filename):
        raise FileNotFoundError("File doesn't exist: {}".format(filename))
    new_ext = ".pbtxt" if is_text else ".pb"
    head, tail = os.path.split(os.path.abspath(filename))
    print("Convert: {} \n     to: {}".format(filename, os.path.join(head, tail + new_ext)))
    graph_def, _, _, _ = load_tf_graph_def(graph_file_name=filename, is_binary=is_text)
    tf_v1.import_graph_def(graph_def, name='')
    tf_v1.train.write_graph(graph_def, head, tail + new_ext, as_text=is_text)


if __name__ == '__main__':
    argv = argparser()
    if argv.pb is None and argv.pbtxt is None:
        print("Please provide model to convert --to_pb or --to_pbtxt")
        sys.exit(1)
    if argv.pb is not None:
        convert(argv.pb, True)
    if argv.pbtxt is not None:
        convert(argv.pbtxt, False)
