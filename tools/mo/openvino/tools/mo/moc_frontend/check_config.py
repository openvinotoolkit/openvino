# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

from openvino.tools.mo.utils import import_extensions
from openvino.tools.mo.utils.error import Error


def any_extensions_used(argv: argparse.Namespace):
    # Checks that extensions are provided.
    # Allowed types are string containing path to legacy extension directory
    # or path to new extension .so file, or classes inherited from BaseExtension.
    if not hasattr(argv, 'extensions') or argv.extensions is None:
        return False

    if isinstance(argv.extensions, list) and len(argv.extensions) > 0:
        has_non_default_path = False
        has_non_str_objects = False
        for ext in argv.extensions:
            if not isinstance(ext, str):
                has_non_str_objects = True
                continue
            if len(ext) == 0 or ext == import_extensions.default_path():
                continue
            has_non_default_path = True

        return has_non_default_path or has_non_str_objects

    raise Exception("Expected list of extensions, got {}.".format(type(argv.extensions)))


def legacy_extensions_used(argv: argparse.Namespace):
    if any_extensions_used(argv):
        extensions = argv.extensions
        legacy_ext_counter = 0
        for extension in extensions:
            if not isinstance(extension, str):
                continue
            if extension == import_extensions.default_path():
                continue
            if not Path(extension).is_file():
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
        extensions = argv.extensions
        if not isinstance(extensions, list):
            extensions = [extensions]
        new_ext_counter = 0
        for extension in extensions:
            if isinstance(extension, str):
                path = Path(extension)
                if path.is_file() and (path.suffix == '.so' or path.suffix == '.dll'):
                    new_ext_counter += 1
            else:
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
        if isinstance(argv.transformations_config, str):
            path = Path(argv.transformations_config)
            if path.is_file():
                return path
    return None


def new_transformations_config_used(argv: argparse.Namespace):
    path = get_transformations_config_path(argv)
    if path != None:
        return is_new_json_config(path)

    if hasattr(argv, 'transformations_config') \
            and argv.transformations_config is not None and not isinstance(argv.transformations_config, str):
        return True

    return False


def legacy_transformations_config_used(argv: argparse.Namespace):
    path = get_transformations_config_path(argv)
    if path != None:
        return not is_new_json_config(path)
    return False


def input_freezig_used(argv):
    return hasattr(argv, 'freeze_placeholder_with_value') and argv.freeze_placeholder_with_value is not None \
        and len(argv.freeze_placeholder_with_value) > 0
