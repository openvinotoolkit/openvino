# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import numbers

from openvino.runtime import get_version as get_rt_version  # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.cli_parser import get_params_with_paths_list
from openvino.tools.ovc.telemetry_params import telemetry_params
from openvino.tools.ovc.utils import check_values_equal

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.ovc.telemetry_stub as tm


def init_mo_telemetry(app_name='Model Conversion API'):
    return init_telemetry_class(tid=get_tid(),
                                app_name=app_name,
                                app_version=get_rt_version(),
                                backend='ga4',
                                enable_opt_in_dialog=False,
                                disable_in_ci=True
                                )


def init_telemetry_class(tid,
                         app_name,
                         app_version,
                         backend,
                         enable_opt_in_dialog,
                         disable_in_ci):
    # Init telemetry class
    telemetry = tm.Telemetry(tid=tid,
                             app_name=app_name,
                             app_version=app_version,
                             backend=backend,
                             enable_opt_in_dialog=enable_opt_in_dialog,
                             disable_in_ci=disable_in_ci)

    # Telemetry is a singleton class and if it was already initialized in another tool
    # some parameters will be incorrect, including app_name.
    # In this case we need to force reinitialisation of telemetry.
    if hasattr(telemetry, "backend") and telemetry.backend.app_name != app_name:
        telemetry.init(tid=tid,
                       app_name=app_name,
                       app_version=app_version,
                       backend=backend,
                       enable_opt_in_dialog=enable_opt_in_dialog,
                       disable_in_ci=disable_in_ci)
    return telemetry


def send_framework_info(framework: str):
    """
    This function sends information about used framework.
    :param framework: framework name.
    """
    t = tm.Telemetry()
    t.send_event('ovc', 'framework', framework)


def get_tid():
    """
    This function returns the ID of the database to send telemetry.
    """
    return telemetry_params['TID']


def send_conversion_result(conversion_result: str, need_shutdown=False):
    t = tm.Telemetry()
    t.send_event('ovc', 'conversion_result', conversion_result)
    t.end_session('ovc')
    if need_shutdown:
        t.force_shutdown(1.0)


def arg_to_str(arg):
    # This method converts to string only known types, otherwise returns string with name of the type
    from openvino.runtime import PartialShape, Shape, Type, Layout  # pylint: disable=no-name-in-module,import-error
    if isinstance(arg, (PartialShape, Shape, Type, Layout)):
        return str(arg)
    if isinstance(arg, (str, numbers.Number, bool)):
        return str(arg)
    return str(type(arg))


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
        if not check_values_equal(arg_value, cli_parser.get_default(arg)):
            if arg in params_with_paths:
                # If command line argument value is a directory or a path to file it is not sent
                # as it may contain confidential information. "1" value is used instead.
                param_str = arg + ":" + str(1)
            else:
                param_str = arg + ":" + arg_to_str(arg_value)

            t.send_event('ovc', 'cli_parameters', param_str)
