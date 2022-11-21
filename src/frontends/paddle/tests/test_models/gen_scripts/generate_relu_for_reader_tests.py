# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from shutil import copyfile
import os
import sys

copyfile(os.path.dirname(os.path.realpath(__file__)) + "/../relu.pdmodel", sys.argv[1] + "relu.pdmodel")
