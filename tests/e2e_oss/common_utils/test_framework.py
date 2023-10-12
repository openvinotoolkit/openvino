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
import re
import random
from time import strftime

import pytest


class FrameworkMessages:
    NOT_IMPLEMENTED = "NOT IMPLEMENTED"
    NEXT_VERSION = "NEXT VERSION"
    NOT_TO_BE_REPORTED_IF_SKIPPED = "NOT TO BE REPORTED IF SKIPPED"
    TWO_USERS_CANT_MOUNT_ON_WINDOWS = "Cannot create mount point by two nctl users on one Windows user"
    TWO_USERS_CANT_MOUNT_ON_NON_LINUX = "Cannot create mount point by two nctl users on one non-Linux user"
    TEST_ISSUE = "TEST ISSUE"
    FEATURE_NOT_READY = "FEATURE NOT READY"
    NOT_SUPPORTED_IN_SAFARI = "Not possible to automate on Safari browser"

class TestStatus:
    RESULT_PASS = "PASS"
    RESULT_FAIL = "FAIL"
    RESULT_SKIPPED = "SKIPPED"
    RESULT_NOT_IMPLEMENTED = FrameworkMessages.NOT_IMPLEMENTED.replace(" ", "_")
    RESULT_NEXT_VERSION = FrameworkMessages.NEXT_VERSION.replace(" ", "_")
    RESULT_UNKNOWN = "UNKNOWN"
    RESULT_NOT_TO_BE_REPORTED = FrameworkMessages.NOT_TO_BE_REPORTED_IF_SKIPPED.replace(" ", "_")
    RESULT_TEST_ISSUE = FrameworkMessages.TEST_ISSUE.replace(" ", "_")
    RESULT_FEATURE_NOT_READY = FrameworkMessages.FEATURE_NOT_READY.replace(" ", "_")

def skip_if_runtime(condition):
    if condition:
        pytest.skip(msg=FrameworkMessages.NOT_TO_BE_REPORTED_IF_SKIPPED)


def skip_if(condition):
    return pytest.mark.skipif(condition, reason=FrameworkMessages.NOT_TO_BE_REPORTED_IF_SKIPPED)


def skip_not_implemented():
    return pytest.mark.skip(reason=FrameworkMessages.NOT_IMPLEMENTED)

def get_short_date_string():
    date_str = strftime("%Y%m%d%H%M%S")
    return date_str


def get_xdist_worker_string():
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def is_xdist_master():
    return get_xdist_worker_string() == "master"


def current_pytest_test_case(separator="_"):
    test_case = os.environ.get("PYTEST_CURRENT_TEST")
    test_case = test_case.split("::")[-1]
    test_case = re.sub('[^a-z0-9]+', separator, test_case)
    return test_case


def generate_test_object_name(separator="_", prefix="tmp", short=False):
    random_sha = hex(random.getrandbits(128))[2:8]
    date_str = get_short_date_string()
    name = separator.join([prefix, os.environ["USER"], date_str, random_sha])
    return name

def get_xdist_worker_nr() -> int:
    xdist_current_worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    if xdist_current_worker == "master":
        xdist_current_worker = 0
    else:
        xdist_current_worker = int(xdist_current_worker.lstrip("gw"))
    return xdist_current_worker

def get_xdist_worker_count() -> int:
    return int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
