#
# INTEL CONFIDENTIAL
# Copyright (c) 2022 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
#
import os

from common import config

""" TT_PRODUCT_VERSION_SUFFIX - Environment version suffix provided by user"""
product_version_suffix = os.environ.get("TT_PRODUCT_VERSION_SUFFIX", "e2e_tests")
config.product_version_suffix = product_version_suffix

""" TT_REPOSITORY_NAME - repository name provided by user """
repository_name = os.environ.get("TT_REPOSITORY_NAME", "openvino.test:e2e_tests")
config.repository_name = repository_name
