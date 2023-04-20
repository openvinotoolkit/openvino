# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from logger_test_actual import create_tf_model


@patch('openvino.tools.mo.convert_impl.driver', side_effect=Exception('MESSAGE'))
def run_main(mock_driver):
    from openvino.tools.mo import convert_model
    convert_model(create_tf_model(), silent=False)


if __name__ == "__main__":
    run_main()
