# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import os
import sys

import logging as log


def ngraph_emit_ir(nGraphFunction, argv: argparse.Namespace):
    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()

    orig_model_name = os.path.normpath(os.path.join(output_dir, argv.model_name))
    nGraphFunction.serialize(orig_model_name + ".xml", orig_model_name + ".bin")
    print('[ SUCCESS ] Converted with nGraph Serializer')
    return 0
