"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import importlib
import logging as log
import os
import pkgutil
import sys

from mo.back.replacement import BackReplacementPattern
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.utils.class_registration import _check_unique_ids, update_registration, get_enabled_and_disabled_transforms
from mo.utils.model_analysis import AnalyzeAction


def import_by_path(path: str, middle_names: list = ()):
    for module_loader, name, ispkg in pkgutil.iter_modules([path]):
        importlib.import_module('{}.{}'.format('.'.join(middle_names), name))


def default_path():
    EXT_DIR_NAME = 'extensions'
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, EXT_DIR_NAME))


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

    front_classes = get_front_classes()
    internal_dirs = {
                         ('ops', ): [Op],
                         ('analysis',): [AnalyzeAction],
                         ('front', ): front_classes,
                         ('front', framework): front_classes,
                         ('middle', ): [MiddleReplacementPattern],
                         ('back', ): [BackReplacementPattern]}

    if ext == 'mo':
        internal_dirs[('front', framework, 'extractors')] = front_classes

    for p in internal_dirs.keys():
        import_by_path(os.path.join(path, *p), [ext, *p])
        update_registration(internal_dirs[p], enabled_transforms, disabled_transforms)
    sys.path.remove(root_dir)


def load_dirs(framework: str, dirs: list, get_front_classes: callable):
    if dirs is None:
        return

    mo_inner_extensions = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'mo'))
    dirs.insert(0, mo_inner_extensions)
    dirs = [os.path.abspath(e) for e in dirs]
    if default_path() not in dirs:
        dirs.insert(0, default_path())
    for path in dirs:
        load_dir(framework, path, get_front_classes)

    _check_unique_ids()
