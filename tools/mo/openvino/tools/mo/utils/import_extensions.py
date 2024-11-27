# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging as log
import os
import pkgutil
import sys

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.load.loader import Loader
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.class_registration import _check_unique_ids, update_registration, \
    get_enabled_and_disabled_transforms, clear_registered_classes_dict
from openvino.tools.mo.utils.model_analysis import AnalyzeAction


def get_internal_dirs(framework: str, get_front_classes: callable):
    front_classes = get_front_classes()
    return {
            ('ops', ): [Op],
            ('analysis',): [AnalyzeAction],
            ('load', framework): [Loader],
            ('front', ): front_classes,
            ('front', framework): front_classes,
            ('front', framework, 'extractors'): front_classes,
            ('middle', ): [MiddleReplacementPattern],
            ('back', ): [BackReplacementPattern]}

def import_by_path(path: str, middle_names: list = (), prefix: str = ''):
    for module_loader, name, ispkg in pkgutil.iter_modules([path]):
        importlib.import_module('{}{}.{}'.format(prefix, '.'.join(middle_names), name))


def default_path():
    EXT_DIR_NAME = '.'
    return os.path.abspath(os.getcwd().join(EXT_DIR_NAME))


def load_dir(framework: str, path: str, get_front_classes: callable):
    """
    Assuming the following sub-directory structure for path:

        front/
            <framework>/
                <other_files>.py
            <other_directories>/
            <other_files>.py
        ops/
            <ops_files>.py
        middle/
            <other_files>.py
        back/
            <other_files>.py

    This function loads modules in the following order:
        1. ops/<ops_files>.py
        2. front/<other_files>.py
        3. front/<framework>/<other_files>.py
        4. middle/<other_files>.py
        5. back/<other_files>.py

    Handlers loaded later override earlier registered handlers for an op.
    1, 2, 3 can concur for the same op, but 4 registers a transformation pass
    and it shouldn't conflict with any stuff loaded by 1, 2 or 3.
    It doesn't load files from front/<other_directories>
    """
    log.info("Importing extensions from: {}".format(path))
    root_dir, ext = os.path.split(path)
    sys.path.insert(0, root_dir)

    enabled_transforms, disabled_transforms = get_enabled_and_disabled_transforms()

    internal_dirs = get_internal_dirs(framework, get_front_classes)
    prefix = 'openvino.tools.' if ext == 'mo' else ''

    exclude_modules = {'tf', 'onnx', 'kaldi', 'caffe'}
    exclude_modules.remove(framework)

    for p in internal_dirs.keys():
        import_by_path(os.path.join(path, *p), [ext, *p], prefix)
        update_registration(internal_dirs[p], enabled_transforms, disabled_transforms, exclude_modules)
    sys.path.remove(root_dir)


def load_dirs(framework: str, dirs: list, get_front_classes: callable):
    if dirs is None:
        return
    internal_dirs = get_internal_dirs(framework, get_front_classes)

    for p, dir_names in internal_dirs.items():
        for d in dir_names:
            d.registered_cls = []
            d.registered_ops = {}
    clear_registered_classes_dict()

    mo_inner_extensions = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'mo'))
    dirs.insert(0, mo_inner_extensions)
    dirs = [os.path.abspath(e) for e in dirs]
    if default_path() not in dirs:
        dirs.insert(0, default_path())
    for path in dirs:
        load_dir(framework, path, get_front_classes)

    _check_unique_ids()
