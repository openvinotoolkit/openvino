# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


""" Fields for logger """
import os
import re

from .core import get_bool, get_list, get_path


class StrippingLists:
    DEFAULT_SENSITIVE_KEYS_TO_BE_MASKED = [
        r"(?!zabbix_operator_initial_).*pass(word)?", r".*client_id", r".*(access)?(_)?(?<!ssh_)key(?!s|_path)",
        r"id_token",
        r"Authorization",
        r"database_url",
        r"gmail_"
    ]


# OpenVINO common parameters
_product_version = os.environ.get("PRODUCT_VERSION", None)
_product_type = os.environ.get("PRODUCT_TYPE", None)
_package_version = os.environ.get("PACKAGE_VERSION", None)

openvino_root_dir = get_path("OPENVINO_ROOT_DIR")
###

host_os_user = os.environ.get("TT_HOST_OS_USER", "root")
log_username = os.environ.get("TT_LOG_USERNAME", False)
run_performance_tests = get_bool("TT_PERFORMANCE_TESTS", False)
logger_format = "{}%(asctime)s {}- %(threadName)s:%(name)s:%(funcName)s:%(lineno)d - %(levelname)s: %(message)s".format if run_performance_tests else \
    "{}%(asctime)s {}- %(name)s - %(levelname)s: %(message)s".format
sensitive_keys_to_be_masked = re.compile(
    "|".join(get_list("TT_SENSITIVE_KEYS", fallback=StrippingLists.DEFAULT_SENSITIVE_KEYS_TO_BE_MASKED)), re.IGNORECASE)
strip_sensitive_data = get_bool("TT_STRIP_SENSITIVE_DATA", False)
logging_level = os.environ.get("TT_LOGGING_LEVEL", "INFO")

"""TT_DATABASE_URL - report database name.  if specified, results will be logged in this database"""
# Workaround for no TT_DATABASE_URL set in Jenkins Job.
database_url = os.environ.get("TT_DATABASE_URL", None)  # if specified, results will be logged in database

"""TT_DATABASE_SSL - use ssl to access database """
database_ssl = get_bool("TT_DATABASE_SSL", False)  # if specified, results will be logged in database

""" TT_CONFIGURATION - configuration name used to distinct test builds. Default value: "" """
configuration = os.environ.get("TT_CONFIGURATION", "")

""" TT_BRANCH_NAME - branch name. Default value is current branch  """
branch_name = os.environ.get("TT_BRANCH_NAME", "")

""" TT_COMMIT_ID - commit id. Default value is current commit id  """
commit_id = os.environ.get("TT_COMMIT_ID", "")

""" TT_STREAM - stream name used to group many test builds. Default value: default """
stream = os.environ.get("TT_STREAM", "default")

""" TT_PRODUCT_VERSION - Environment version provided by user"""
product_version = os.environ.get("TT_PRODUCT_VERSION", _product_version)

""" TT_PRODUCT_BUILD_NUMBER  - Test product build number provided by user (last number from version - 0.8.0.XXXX)"""

if _product_version and _product_type and _package_version:
    _product_build_number_default = f"{_product_type}_{_product_version}_{_package_version}"
else:
    _product_build_number_default = "Unset_product_build_number"
product_build_number = os.environ.get("TT_PRODUCT_BUILD_NUMBER", None)

""" TT_PRODUCT_VERSION_SUFFIX - Environment version suffix provided by user"""
product_version_suffix = os.environ.get("TT_PRODUCT_VERSION_SUFFIX", "")

""" TT_INFO_MODULE - indicates module that should be used for getting information about tested environment.
                     Default value: e2e.base_info.BaseInfo.
                     Allowed value class module that inherits from default class"""
info_module = os.environ.get("TT_INFO_MODULE", "e2e_tests.common.environment_info.BaseInfo")

""" TT_REPOSITORY_NAME - repository name provided by user """
repository_name = os.environ.get("TT_REPOSITORY_NAME", "")

"""TT_BUILD_URL - Link to build where tests are being executed"""
test_build_log_url = os.environ.get("TT_BUILD_URL", "")

""" TT_TEST_RUN_ID - id of test run document. Set if you want to place your test run to existing document.
                        Make sure collection contains document with given id"""
test_run_id = os.environ.get("TT_TEST_RUN_ID", None)

""" TT_TEST_SESSION_BUILD_NUMBER  - Test session build number provided by user or CI"""
test_session_build_number = os.environ.get("TT_TEST_SESSION_BUILD_NUMBER", "0.0")

""" TT_ENVIRONMENT_NAME - Environment name to be used while reporting test results
                          to be presented on test reports as a environment name."""
environment_name = os.environ.get("TT_ENVIRONMENT_NAME", "")

""" TT_BUGS - Filter collected test cases by provided bugs list, only tests marked by 
              @pytest.mark.bugs(...) will be collected (and executed). """
bug_ids = os.environ.get("TT_BUGS", None)

""" TT_REQUIREMENTS - Filter collected test cases by provided requirements list. Only tests marked by 
                      @pytest.mark.reqids(...) will be collected (and executed). """
req_ids = os.environ.get("TT_REQUIREMENTS", None)

""" TT_COMPONENTS - Filter collected test cases by provided components list. Only tests marked by 
                    @pytest.mark.components(...) will be collected (and executed)."""
components_ids = os.environ.get("TT_COMPONENTS", None)

"""TT_ON_COMMIT_TESTS - False -> api-on-commit tests are not run,
                             True -> api-on-commit tests are run, default: False"""
run_on_commit_tests = get_bool("TT_ON_COMMIT_TESTS", True)

"""TT_RUN_REGRESSION_TESTS - False -> api-regression tests are not run,
                             True -> api-regression tests are run, default: False"""
run_regression_tests = get_bool("TT_RUN_REGRESSION_TESTS", True)

"""TT_RUN_ENABLING_TESTS - False -> api-enabling tests are not run,
                             True -> api-enabling tests are run, default: False"""
run_enabling_tests = get_bool("TT_ENABLING_TESTS", True)
