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

base = "openvino"
name = "runtime"

module_name = "{}.{}".format(__name__, name)
export_module_name = "{}.{}".format(base, name)
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
