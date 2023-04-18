# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from logger_test_actual import create_tf_model


def mocked_check_module_import(module_name, required_version, key, not_satisfied_versions):
    if module_name == 'importlib-metadata':
        raise ImportError()


# Patch check_module_import to have unsatisfied dependency
@patch('openvino.tools.mo.utils.versions_checker.check_module_import', mocked_check_module_import)
def run_main():
    from openvino.tools.mo import convert_model

    # convert_model() should successfully convert, but not show unsatisfied dependency
    convert_model(create_tf_model(), silent=False)


if __name__ == "__main__":
    run_main()
