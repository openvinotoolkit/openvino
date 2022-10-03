#!/usr/bin/env python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import platform
import sys

try:
    import openvino.tools.mo
    execution_type = "mo"
except ModuleNotFoundError:
    mo_root_path = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    sys.path.insert(0, mo_root_path)
    execution_type = "install_prerequisites.{}".format("bat" if platform.system() == "Windows" else "sh")

import openvino.tools.mo.utils.version as v
try:
    import openvino_telemetry as tm  # pylint: disable=import-error,no-name-in-module
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm
from openvino.tools.mo.utils.error import classify_error_type
from openvino.tools.mo.utils.telemetry_utils import get_tid


def send_telemetry(mo_version: str, message: str, event_type: str):
    t = tm.Telemetry(tid=get_tid(), app_name='Version Checker', app_version=mo_version)
    # do not trigger new session if we are executing from the check from within the MO because it is actually not model
    # conversion run which we want to send
    if execution_type != 'mo':
        t.start_session(execution_type)
    t.send_event(execution_type, event_type, message)
    if execution_type != "mo":
        t.end_session(execution_type)
    t.force_shutdown(1.0)


def import_core_modules(silent: bool, path_to_module: str):
    """
        This function checks that OpenVINO Python API is available
        and necessary python modules exists. So the next list of imports
        must contain all IE/NG Python API imports that are used inside MO.

    :param silent: enables or disables logs printing to stdout
    :param path_to_module: path where python API modules were found
    :return: True if all imports were successful and False otherwise
    """
    try:
        from openvino._offline_transformations import apply_moc_transformations, apply_moc_legacy_transformations,\
            apply_low_latency_transformation, apply_fused_names_cleanup  # pylint: disable=import-error,no-name-in-module
        from openvino._offline_transformations import apply_make_stateful_transformation, generate_mapping_file  # pylint: disable=import-error,no-name-in-module
        from openvino._offline_transformations import generate_mapping_file, apply_make_stateful_transformation  # pylint: disable=import-error,no-name-in-module

        from openvino.runtime import Model, serialize, get_version  # pylint: disable=import-error,no-name-in-module
        from openvino.runtime.op import Parameter  # pylint: disable=import-error,no-name-in-module
        from openvino.runtime import PartialShape, Dimension  # pylint: disable=import-error,no-name-in-module
        from openvino.frontend import FrontEndManager, FrontEnd  # pylint: disable=no-name-in-module,import-error

        import openvino.frontend  # pylint: disable=import-error,no-name-in-module

        if silent:
            return True

        ie_version = str(get_version())
        mo_version = str(v.get_version())  # pylint: disable=no-member,no-name-in-module

        print("{}: \t{}".format("OpenVINO runtime found in", os.path.dirname(openvino.__file__)))
        print("{}: \t{}".format("OpenVINO runtime version", ie_version))
        print("{}: \t{}".format("Model Optimizer version", mo_version))

        versions_mismatch = False

        mo_hash = v.extract_hash_from_version(mo_version)
        ie_hash = v.extract_hash_from_version(ie_version)

        if mo_hash is not None and ie_hash is not None:
            min_length = min(len(mo_hash), len(ie_hash))
            mo_hash = mo_hash[:min_length]
            ie_hash = ie_hash[:min_length]

        if mo_hash != ie_hash or mo_hash is None or ie_hash is None:
            versions_mismatch = True
            extracted_mo_release_version = v.extract_release_version(mo_version)
            mo_is_custom = extracted_mo_release_version == (None, None)

            print("[ WARNING ] Model Optimizer and OpenVINO runtime versions do not match.")
            print("[ WARNING ] Consider building the OpenVINO Python API from sources or reinstall OpenVINO "
                  "(TM) toolkit using", end=" ")
            if mo_is_custom:
                print("\"pip install openvino\" (may be incompatible with the current Model Optimizer version)")
            else:
                print("\"pip install openvino=={}.{}\"".format(*extracted_mo_release_version))

        simplified_mo_version = v.get_simplified_mo_version()
        message = str(dict({
            "platform": platform.system(),
            "mo_version": simplified_mo_version,
            "ie_version": v.get_simplified_ie_version(version=ie_version),
            "versions_mismatch": versions_mismatch,
        }))
        send_telemetry(simplified_mo_version, message, 'ie_version_check')

        return True
    except Exception as e:
        # Do not print a warning if module wasn't found or silent mode is on
        if "No module named 'openvino" not in str(e):
            print("[ WARNING ] Failed to import OpenVINO Python API in: {}".format(path_to_module))
            print("[ WARNING ] {}".format(e))

            # Send telemetry message about warning
            simplified_mo_version = v.get_simplified_mo_version()
            message = str(dict({
                "platform": platform.system(),
                "mo_version": simplified_mo_version,
                "ie_version": v.get_simplified_ie_version(env=os.environ),
                "python_version": sys.version,
                "error_type": classify_error_type(e),
            }))
            send_telemetry(simplified_mo_version, message, 'ie_import_failed')

        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--path_to_module")
    args = parser.parse_args()

    if not import_core_modules(args.silent, args.path_to_module):
        exit(1)
