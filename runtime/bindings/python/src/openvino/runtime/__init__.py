# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("openvino-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"

import os
import importlib
import sys


def __load_module_as_parent_module(base, name):
    module_name = "{}.{}".format(__name__, name)
    export_module_name = "{}".format(base)
    native_module = sys.modules.pop(module_name, None)
    try:
        py_module = importlib.import_module(module_name)
    except ImportError as err:
        print("Can't load Python code for module:", module_name,
                  ". Reason:", err)

    if not hasattr(base, name):
        setattr(sys.modules[base], name, py_module)
    sys.modules[export_module_name] = py_module
    # If it is C extension module it is already loaded by cv2 package
    if native_module:
        setattr(py_module, "_native", native_module)
        for k, v in filter(lambda kv: not hasattr(py_module, kv[0]),
                           native_module.__dict__.items()):
            print('    symbol({}): {} = {}'.format(name, k, v))
            setattr(py_module, k, v)


def __collect_extra_submodules(enable_debug_print=False):
    def modules_filter(module):
        return all((
             # module is not internal
             not module.startswith("_"),
             not module.startswith("python-"),
             # it is not a file
             os.path.isdir(os.path.join(_extra_submodules_init_path, module))
        ))
    if sys.version_info[0] < 3:
        if enable_debug_print:
            print("Extra submodules is loaded only for Python 3")
        return []

    __INIT_FILE_PATH = os.path.abspath(__file__)
    _extra_submodules_init_path = os.path.dirname(__INIT_FILE_PATH)
    return filter(modules_filter, os.listdir(_extra_submodules_init_path))


# for submodule in __collect_extra_submodules():
#     __load_module_as_parent_module("openvino", submodule)
__load_module_as_parent_module("openvino", "runtime")
