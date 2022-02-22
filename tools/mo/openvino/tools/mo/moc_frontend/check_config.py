# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

from openvino.tools.mo.utils import import_extensions
from openvino.tools.mo.utils.error import Error


def any_extensions_used(argv: argparse.Namespace):
    return hasattr(argv, 'extensions') and argv.extensions is not None and len(argv.extensions) > 0 \
        and argv.extensions != import_extensions.default_path() # extensions arg has default value


def legacy_extensions_used(argv: argparse.Namespace):
    if any_extensions_used(argv):
        extensions = argv.extensions.split(',')
        legacy_ext_counter = 0
        for extension in extensions:
            path = Path(extension)
            if not path.is_file():
                legacy_ext_counter += 1
        if legacy_ext_counter == len(extensions):
            return True # provided only legacy extensions
        elif legacy_ext_counter == 0:
            return False # provided only new extensions
        else:
            raise Error('Using new and legacy extensions in the same time is forbidden')
    return False


def new_extensions_used(argv: argparse.Namespace):
    if any_extensions_used(argv):
        extensions = argv.extensions.split(',')
        new_ext_counter = 0
        for extension in argv.extensions.split(','):
            path = Path(extension)
            if path.is_file() and (path.suffix == '.so' or path.suffix == '.dll'):
                new_ext_counter += 1
        if new_ext_counter == len(extensions):
            return True # provided only new extensions
        elif new_ext_counter == 0:
            return False # provided only legacy extensions
        else:
            raise Error('Using new and legacy extensions in the same time is forbidden')
    return False


def is_new_json_config(json_file_path: str):
    with open(json_file_path) as stream:
        config_content = json.load(stream)
        if len(config_content) == 0: # empty case
            return False
        if isinstance(config_content, dict): # single transformation
            return 'library' in config_content.keys()
        # many transformations in single file
        library_counter = 0
        for transform in config_content:
            if any(key == 'library' for key in transform.keys()):
                library_counter+=1
        if len(config_content) == library_counter: # all transformations has 'library' attribute
            return True
        elif library_counter == 0: # all transformations are legacy type
            return False
        else:
            raise Error('Mixed types of transformations configurations were used')


def get_transformations_config_path(argv: argparse.Namespace) -> Path:
    if hasattr(argv, 'transformations_config') \
        and argv.transformations_config is not None and len(argv.transformations_config):
        path = Path(argv.transformations_config)
        if path.is_file():
            return path
    return None


def new_transformations_config_used(argv: argparse.Namespace):
    path = get_transformations_config_path(argv)
    if path != None:
        return is_new_json_config(path)
    return False


def legacy_transformations_config_used(argv: argparse.Namespace):
    path = get_transformations_config_path(argv)
    if path != None:
        return not is_new_json_config(path)
    return False


def input_freezig_used(argv):
    return hasattr(argv, 'freeze_placeholder_with_value') and argv.freeze_placeholder_with_value is not None \
        and len(argv.freeze_placeholder_with_value) > 0
