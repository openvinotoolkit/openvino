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
import pathlib
import re
import sys
from copy import copy

from cpuinfo import get_cpu_info

try:
    # In user_config.py, user might export custom environment variables
    from . import user_config
    print("Successfully imported user_config")
except ImportError:
    pass

from . import config    # init config variables

from common_utils.sys_info_utils import get_sys_info

from common_utils.ir_providers.tf_helper import TFVersionHelper
from common_utils.logger import get_logger
from common_utils.hook_utils import send_results_to_validation_report

from e2e_oss.plugins.common.conftest import *

NODEID_TOKENS_RE = r"(?P<file>.+?)::(?P<func_name>.+?)\[(?P<args>.+?)\]"
VR_FRIENDLY_NODEID = "{class_definition_path}::{class_name}::{func_name}[{args}]"

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
logger = get_logger(__name__)


def pytest_configure(config):
    sys_info = get_sys_info()
    cpu_info = get_cpu_info()

    logger.info(f"System information: {sys_info}")
    logger.info(f"CPU info: {cpu_info}")
    # Fill environment section of HTML report with additional data
    # config._metadata['INTERNAL_GFX_DRIVER_VERSION'] = os.getenv('INTERNAL_GFX_DRIVER_VERSION')
    # Set TensorFlow models version with command line option value
    tf_models_version = config.getoption("tf_models_version")
    TFVersionHelper(tf_models_version)


def _get_vr_friendly_item(item):
    """
    Description:
        Transform item.nodeid to more descriptive form that can be expanded in Validation Report.
        Expanded names includes:
        - Class definition path in openvino.tests repository
        - Explicit class name in pytest format
        - Execution settings.
    Example of vr friendly name:
        'pipelines/production/onnx/winml/light/candy.py::AI_WinML_Candy::test_run[api_2_False_batch_1_device_CPU_precision_FP32]'
    """
    try:
        tokens = re.match(NODEID_TOKENS_RE, item.nodeid).groupdict()
        tokens['class_name'] = type(item.callspec.params['instance']).__name__
        tokens['class_definition_path']  = item.callspec.params['instance'].definition_path
        friendly_name = VR_FRIENDLY_NODEID.format(**tokens)
        item_copy = copy(item)
        item_copy._nodeid = friendly_name
        return item_copy
    except Exception as e:
        logger.error(str(e))
    return item

@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    pytest_html = item.config.pluginmanager.getplugin('html')
    report = (yield).get_result()
    extra = getattr(report, 'extra', [])
    ir_links = []
    if report.when == 'call':
        ir_link = next((p[1] for p in report.user_properties if p[0] == "ir_link"), None)
        if ir_link:
            extra.append(pytest_html.extras.url(ir_link, name="xml"))
            extra.append(pytest_html.extras.url(ir_link.replace(".xml", ".bin"), name="bin"))
            extra.append(pytest_html.extras.url(ir_link.replace(".xml", ".mo_log.txt"), name="mo_log"))

            ir_links.append(f"<a class=\"url\" href=\"{ir_link}\" target=\"_blank\">xml</a>")
            ir_links.append(f"<a class=\"url\" href=\"{ir_link.replace('.xml', '.bin')}\" target=\"_blank\">bin</a>")
            ir_links.append(f"<a class=\"url\" href=\"{ir_link.replace('.xml', '.mo_log.txt')}\" "
                            f"target=\"_blank\">mo_log</a>")
        if getattr(item._request, 'test_info', None):
            item._request.test_info.update(
                {"links": " ".join(ir_links),
                 "log": "\n\n\n".join([report.caplog, report.longreprtext]),
                 "insertTime": report.duration,
                 "duration": report.duration,
                 "result": report.outcome}
            )
    report.extra = extra

    vr_friendly_item = _get_vr_friendly_item(item)
    send_results_to_validation_report(vr_friendly_item, report)
