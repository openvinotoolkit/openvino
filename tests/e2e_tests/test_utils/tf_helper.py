# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-member

"""
The module contains helper class for working with TensorFlow version dependencies
"""
import os
import logging as log
from distutils.version import LooseVersion

import yaml


class TFVersionHelper:
    """
    Contain version of used TensorFlow models for IR conversion

    Description: Some TensorFlow IR providers and test classes use TF version as a part of
    auxiliary file paths but the current TensorFlow version may not match the version needed
    """

    _instance = None

    _configs_map = None
    __tf_models_version = None

    def __new__(cls, *_args, **_kwargs):
        """Singleton
        We consider having one TensorFlow models version per session. Once created the object is
        stored as _instance and shall be returned then in case of attempt to create new object.
        """
        if not TFVersionHelper._instance:
            TFVersionHelper._instance = super(TFVersionHelper, cls).__new__(cls)
        return TFVersionHelper._instance

    def __init__(self, tf_models_version: str = None):
        """Set TF models version as explicit value or installed TF version"""
        if self._configs_map:
            return
        if tf_models_version is None:
            try:
                # the following try-catch is useful to be able to run non-tensorflow tests
                # without the need to have tensorflow installed
                import tensorflow as tf

                tf_models_version = tf.__version__
            except ImportError:
                log.warning("Module 'tensorflow' is not found")
        else:
            log.info(
                'Version of TensorFlow models has been changed to "%s" explicitly',
                tf_models_version,
            )
        with open(
            os.path.join(os.path.dirname(__file__), "tf_helper_config.yml"), "r"
        ) as configs_map_file:
            self._configs_map = yaml.safe_load(configs_map_file)
        self.__tf_models_version = tf_models_version

    @property
    def tf_models_version(self):
        """ Return defined TF models version """
        if self.__tf_models_version is None:
            raise AttributeError("attribute 'tf_models_version' is not defined!")
        return self.__tf_models_version

    def _get_transformations_config_file_name(self, model_type: str, config_versions: list):
        """
        Return sub-graph replacement config file name for models based on TF version and models type
        """
        tf_models_loose_version = LooseVersion(self.__tf_models_version)
        for version in config_versions:
            if tf_models_loose_version >= LooseVersion(str(version)):
                return f"{model_type}_support_api_v{version}.json"
        if model_type == "ssd":
            return "ssd_v2_support.json"
        return f"{model_type}_support.json"

    def resolve_tf_transformations_config(self, config_alias: str, relative_mo: bool = False):
        """Return name of sub-graph replacement config file or its path relative MO root folder"""
        config_info = self._configs_map.get(config_alias)
        if config_info:
            config_file_name = self._get_transformations_config_file_name(
                config_info["model_type"], config_info["versions"]
            )
            return f"front/tf/{config_file_name}" if relative_mo else config_file_name
        return config_alias
