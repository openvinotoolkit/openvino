# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from openvino.tools.mo.utils.error import Error
import os


def default_path():
    EXT_DIR_NAME = '.'
    return os.path.abspath(os.getcwd().join(EXT_DIR_NAME))


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
            if len(ext) == 0 or ext == default_path():
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
            if extension == default_path():
                continue
            if not Path(extension).is_file():
                legacy_ext_counter += 1
        if legacy_ext_counter == len(extensions):
            return True  # provided only legacy extensions
        elif legacy_ext_counter == 0:
            return False  # provided only new extensions
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
            return True  # provided only new extensions
        elif new_ext_counter == 0:
            return False  # provided only legacy extensions
        else:
            raise Error('Using new and legacy extensions in the same time is forbidden')
    return False


def get_transformations_config_path(argv: argparse.Namespace) -> Path:
    if hasattr(argv, 'transformations_config') \
            and argv.transformations_config is not None and len(argv.transformations_config):
        if isinstance(argv.transformations_config, str):
            path = Path(argv.transformations_config)
            if path.is_file():
                return path
    return None


def legacy_transformations_config_used(argv: argparse.Namespace):
    return get_transformations_config_path(argv) != None


def tensorflow_custom_operations_config_update_used(argv: argparse.Namespace):
    return hasattr(argv, 'tensorflow_custom_operations_config_update') and \
           argv.tensorflow_custom_operations_config_update is not None


def input_freezig_used(argv):
    return hasattr(argv, 'freeze_placeholder_with_value') and argv.freeze_placeholder_with_value is not None \
           and len(argv.freeze_placeholder_with_value) > 0
