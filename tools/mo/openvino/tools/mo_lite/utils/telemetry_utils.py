# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse

from openvino.tools.mo_lite.utils.cli_parser import get_params_with_paths_list
from openvino.tools.mo_lite.utils.telemetry_params import telemetry_params
from openvino.tools.mo_lite.utils.version import get_simplified_mo_version

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo_lite.utils.telemetry_stub as tm


def init_mo_telemetry():
    _ = tm.Telemetry(tid=get_tid(), app_name='Model Optimizer', app_version=get_simplified_mo_version())


def send_params_info(argv: argparse.Namespace, cli_parser: argparse.ArgumentParser):
    """
    This function sends information about used command line parameters.
    :param argv: command line parameters.
    :param cli_parser: command line parameters parser.
    """
    t = tm.Telemetry()
    params_with_paths = get_params_with_paths_list()
    for arg in vars(argv):
        arg_value = getattr(argv, arg)
        if arg_value != cli_parser.get_default(arg):
            if arg in params_with_paths:
                # If command line argument value is a directory or a path to file it is not sent
                # as it may contain confidential information. "1" value is used instead.
                param_str = arg + ":" + str(1)
            else:
                param_str = arg + ":" + str(arg_value)

            t.send_event('mo', 'cli_parameters', param_str)


def send_framework_info(framework: str):
    """
    This function sends information about used framework.
    :param framework: framework name.
    """
    t = tm.Telemetry()
    t.send_event('mo', 'framework', framework)


def get_tid():
    """
    This function returns the ID of the database to send telemetry.
    """
    return telemetry_params['TID']
