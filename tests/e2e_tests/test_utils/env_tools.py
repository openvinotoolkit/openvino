# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


class EnvironmentConfigException(Exception):
    """ Environment configuration exception """


class Environment:
    """
    Environment used by tests.

    :attr env:  environment dictionary. populated dynamically from environment
                configuration file.
    """

    env = {}
    locked_dirs = []

    @classmethod
    def abs_path(cls, env_key, *paths):
        """Construct absolute path by appending paths to environment value.

        :param cls: class
        :param env_key: Environment.env key used to get the base path
        :param paths:   paths to be appended to Environment.env value
        :return:    absolute path string where Environment.env[env_key] is
                    appended with paths
        """
        if not cls.env:
            raise EnvironmentConfigException(
                "Test environment is not initialized. "
                "Please initialize environment by calling `fix_env_conf` function before usage."
            )

        if env_key not in cls.env:
            raise EnvironmentConfigException(
                f"Key {env_key} is absent in environment dictionary: {cls.env}\n"
                f"Please check environment configuration file."
            )

        return str(Path(cls.env[env_key], *paths))
