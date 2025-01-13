# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=import-error,logging-fstring-interpolation,fixme

"""
The module implements OpenVINOResources class which provide interface for getting paths to various
OpenVINO resources (tools, samples, etc) according to product installation layout.
"""

import logging
import os
import subprocess

from pathlib import Path
from distutils import spawn

from e2e_tests.common.config import openvino_root_dir
from e2e_tests.common.sys_info_utils import os_type_is_windows


class OpenVINOResourceNotFound(Exception):
    """OpenVINO resource not found exception"""


class OpenVINOResources:
    """Class for getting paths to OpenVINO resources"""

    _resources = {}
    _instance = None

    def __new__(cls, *_args, **_kwargs):
        """Singleton"""
        if not OpenVINOResources._instance:
            OpenVINOResources._instance = super(OpenVINOResources, cls).__new__(cls)
        return OpenVINOResources._instance

    def __init__(self):
        if self._resources:
            return
        self._log = logging.getLogger(self.__class__.__name__)

    def _check_resource(self, resource_name, resource_path):
        """Save resource with specified name, path to self._resources and return True if resource
        path exists, return False otherwise"""
        if resource_path:
            resource_path = Path(resource_path)
            if resource_path.exists():
                self._resources[resource_name] = resource_path
                self._log.info(f"OpenVINO resource {resource_name} found: {resource_path}")
                return True

        self._log.warning(f"OpenVINO resource {resource_name} not found: {resource_path}")
        return False

    def _get_executable_from_os_path(self, resource_name, resource_filename):
        """Find and return absolute path to resource_name executable from system os PATH"""
        if self._resources.get(resource_name):
            return self._resources[resource_name]

        if self._check_resource(resource_name, spawn.find_executable(str(resource_filename))):
            return self._resources[resource_name]

        raise OpenVINOResourceNotFound(f"OpenVINO resource {resource_name} not found")

    @property
    def setupvars(self):
        """Return absolute path to OpenVINO setupvars.[bat|sh] script"""
        resource_name = "setupvars"

        if self._resources.get(resource_name):
            return self._resources[resource_name]

        setupvars = "setupvars.bat" if os_type_is_windows() else "setupvars.sh"

        if os.getenv("OPENVINO_ROOT_DIR"):
            if self._check_resource(
                resource_name, Path(os.getenv("OPENVINO_ROOT_DIR")) / setupvars
            ):
                return self._resources[resource_name]

        raise OpenVINOResourceNotFound(
            f"OpenVINO resource {resource_name} not found, "
            f"OPENVINO_ROOT_DIR environment variable is not set."
        )

    @property
    def install_openvino_dependencies(self):
        """Return absolute path to OpenVINO install_dependencies/install_openvino_dependencies.sh script"""
        resource_name = "install_openvino_dependencies"

        if openvino_root_dir:
            if self._check_resource(
                resource_name,
                Path(openvino_root_dir)
                / "install_dependencies"
                / "install_openvino_dependencies.sh",
            ):
                return self._resources[resource_name]

        raise OpenVINOResourceNotFound(
            f"OpenVINO resource {resource_name} not found, "
            f"OPENVINO_ROOT_DIR environment variable is not set."
        )

    @property
    def omz_pytorch_to_onnx_converter(self):
        """Return absolute path to omz pytorch to onnx converter"""
        resource_name = "model_loader"

        omz_root_path = self.omz_root
        if self._check_resource(
            resource_name,
            omz_root_path
            / "internal_scripts"
            / "pytorch_to_onnx.py"
        ):
            return self._resources[resource_name]

    @property
    def omz_root(self):
        """Return absolute path to OMZ root directory"""
        resource_name = "omz_root"

        if self._resources.get(resource_name):
            return self._resources[resource_name]

        try:
            # pylint: disable=import-outside-toplevel

            # Import only when really called to avoid import errors when OpenVINOResources is
            # imported but accuracy checker tool is absent on the system.
            from openvino.tools import accuracy_checker

            if self._check_resource(
                resource_name, Path(accuracy_checker.__file__).parents[2] / "model_zoo"
            ):
                return self._resources[resource_name]
        except ImportError as exc:  # pylint: disable=unused-variable
            if os.getenv("OMZ_ROOT"):
                print("OMZ ROOT IS: {}".format(os.getenv("OMZ_ROOT")))
                if self._check_resource(resource_name, Path(os.getenv("OMZ_ROOT"))):
                    return self._resources[resource_name]

        raise OpenVINOResourceNotFound(f"OpenVINO resource {resource_name} not found")

    @property
    def omz_info_dumper(self):
        """Return absolute path to OMZ info_dumper tool"""
        return self._get_executable_from_os_path("omz_info_dumper", "omz_info_dumper")

    @property
    def omz_downloader(self):
        """Return absolute path to OMZ downloader tool"""
        return self._get_executable_from_os_path("omz_downloader", "omz_downloader")

    @property
    def omz_converter(self):
        """Return absolute path to OMZ converter tool"""
        return self._get_executable_from_os_path("omz_converter", "omz_converter")

    @property
    def omz_quantizer(self):
        """Return absolute path to OMZ quantizer tool"""
        return self._get_executable_from_os_path("omz_quantizer", "omz_quantizer")

    @property
    def pot(self):
        """Return absolute path to Post-training Optimization tool (pot)"""
        return self._get_executable_from_os_path("pot", "pot")

    @property
    def pot_speech_sample(self):
        """Return absolute path to POT speech sample  (gna_sample.py)"""
        resource_name = "pot_speech_sample"

        if self._resources.get(resource_name):
            return self._resources[resource_name]

        try:
            # pylint: disable=import-outside-toplevel

            # Import only when really called to avoid import errors when OpenVINOResources is
            # imported but pot tool is absent on the system.
            from openvino import tools
        except ImportError as exc:
            raise OpenVINOResourceNotFound(f"OpenVINO resource {resource_name} not found") from exc

        if self._check_resource(
            resource_name,
            Path(tools.__file__).parent / "pot" / "api" / "samples" / "speech" / "gna_sample.py",
        ):
            return self._resources[resource_name]

        raise OpenVINOResourceNotFound(f"OpenVINO resource {resource_name} not found")

    def add_setupvars_cmd(self, cmd):
        """Return final command line with setupvars script"""

        input_cmd = (
            subprocess.list2cmdline(list(map(str, cmd))) if isinstance(cmd, list) else str(cmd)
        )
        input_cmd_escaped = input_cmd.replace('"', '\\"')

        output_cmd = (
            f"call {self.setupvars} && set && {input_cmd}"
            if os_type_is_windows()
            else f'bash -c ". {self.setupvars} && env && {input_cmd_escaped}"'
        )
        return output_cmd
