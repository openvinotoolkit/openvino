# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common import config

""" TT_PRODUCT_VERSION_SUFFIX - Environment version suffix provided by user"""
product_version_suffix = os.environ.get("TT_PRODUCT_VERSION_SUFFIX", "e2e_tests")
config.product_version_suffix = product_version_suffix

""" TT_REPOSITORY_NAME - repository name provided by user """
repository_name = os.environ.get("TT_REPOSITORY_NAME", "openvino.test:e2e_tests")
config.repository_name = repository_name
