# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib

import distro

from . import config
from .logger import get_logger

logger = get_logger(__name__)

DEFAULT_BUILD_NUMBER = 0
DEFAULT_SHORT_VERSION_NUMBER = "0.0.0"
DEFAULT_FULL_VERSION_NUMBER = "{}-{}-{}".format(DEFAULT_SHORT_VERSION_NUMBER, config.product_version_suffix,
                                                DEFAULT_BUILD_NUMBER)


class BaseInfo:
    """Retrieves environment info"""
    glob_version = None
    glob_os_distname = None

    @property
    def version(self):
        """Retrieves version, but only once.

        If retrieval doesn't work, default version is returned.
        """
        if self.glob_version is None:
            self.glob_version = self.get()
            self.glob_version = \
                self.glob_version["version"]

        return self.glob_version

    @property
    def os_distname(self):
        """Retrieves os distname, but only once."""
        if self.glob_os_distname is None:
            self.glob_os_distname = distro.linux_distribution()[0]

        return self.glob_os_distname

    @classmethod
    def get(cls):
        """
        Returns constant environment info.
        """
        logger.info("BASIC INFO WITHOUT ANY API CALL")
        return {"version": DEFAULT_FULL_VERSION_NUMBER}


class EnvironmentInfo(object):
    """Stores details about environment such as build number, version number
    and allows their retrieval"""
    module_class_string = config.info_module
    module_name, class_name = module_class_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_info = getattr(module, class_name)
    env_info = class_info()

    @classmethod
    def get_build_number(cls):
        """Retrieves build number from the environment info"""
        if config.product_build_number:
            return config.product_build_number
        return DEFAULT_BUILD_NUMBER

    @classmethod
    def get_version_number(cls):
        """Retrieves version number from the environment info"""
        if config.product_version:
            return config.product_version
        return DEFAULT_FULL_VERSION_NUMBER

    @classmethod
    def get_environment_name(cls):
        """Retrieves the environment name that will be reported for a test run"""
        return config.environment_name

    @classmethod
    def get_os_distname(cls):
        """Retrieves the operating system distribution name"""
        return cls.env_info.os_distname

    @classmethod
    def _retrieve_version_number_from_environment(cls):
        return cls._version_number_from_environment_version(cls.env_info.version)

    @classmethod
    def _retrieve_build_number_from_environment(cls):
        return cls._build_number_from_environment_version(cls.env_info.version)

    @classmethod
    def _build_number_from_environment_version(cls, environment_version):
        return environment_version.split("-")[-1]

    @classmethod
    def _version_number_from_environment_version(cls, environment_version):
        return '-'.join(environment_version.split('-')[:2])
