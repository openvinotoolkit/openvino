# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from unittest.mock import patch

from openvino.tools.mo.utils.error import FrameworkError


def mocked_parse_args(*argv):
    # Mock parse_args method which generates warning
    import logging as log
    log.error("warning", extra={'is_warning': True})
    argv = argparse.Namespace(use_legacy_frontend=False,
                              use_new_frontend=False,
                              framework=None,
                              input_model="abc.pbtxt")
    return argv


@patch('argparse.ArgumentParser.parse_args', mocked_parse_args)
@patch('openvino.tools.mo.convert_impl.driver', side_effect=FrameworkError('FW ERROR MESSAGE'))
def run_main(mock_driver):
    from openvino.tools.mo.main import main
    # runs main() method where driver() raises FrameworkError
    main(argparse.ArgumentParser())


if __name__ == "__main__":
    run_main()
