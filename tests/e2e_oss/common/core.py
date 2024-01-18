#
# INTEL CONFIDENTIAL
# Copyright (c) 2021 Intel Corporation
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


def get_list(key_name, delimiter=',', fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        value = value.split(delimiter)
    elif not value:
        value = []
    return value


def get_bool(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        value = value.lower()
        if value == "true":
            value = True
        elif value == "false":
            value = False
        else:
            raise ValueError("Value of {} env variable is '{}'. Should be 'True' or 'False'.".format(key_name, value))
    return value


def get_int(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        try:
            value = int(value)
        except ValueError:
            raise ValueError("Value '{}' of {} env variable cannot be cast to int.".format(value, key_name))
    return value


def get_path(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value:
        value = os.path.expanduser(value)
        value = os.path.realpath(value)
    return value
