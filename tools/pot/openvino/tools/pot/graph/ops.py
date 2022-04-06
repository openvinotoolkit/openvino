# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import os.path
import importlib
import inspect
import openvino.tools.mo.ops


def get_operations_list():
    ops = {}
    for package in [openvino.tools.mo.ops]:
        for file in glob.glob(os.path.join(os.path.dirname(os.path.abspath(package.__file__)), '*.py')):
            name = '.{}'.format(os.path.splitext(os.path.basename(file))[0])
            module = importlib.import_module(name, package.__name__)
            for key in dir(module):
                obj = getattr(module, key)
                if inspect.isclass(obj):
                    op = getattr(obj, 'op', None)
                    if op is not None:
                        ops[op] = obj
    return ops


OPERATIONS = get_operations_list()
