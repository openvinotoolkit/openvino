# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.ovc.convert import convert_model
from openvino.tools.ovc.telemetry_utils import is_optimum, init_ovc_telemetry

import importlib.metadata as importlib_metadata

try:
    optimum_version = importlib_metadata.version("optimum-intel")
except importlib_metadata.PackageNotFoundError:
    optimum_version = None

from openvino.runtime import get_version as get_rt_version  # pylint: disable=no-name-in-module,import-error
telemetry = init_ovc_telemetry('OpenVINO')
telemetry.send_event("ov", "import", "general_import")

if is_optimum() and optimum_version is not None:
    telemetry = init_ovc_telemetry("Optimum Intel", optimum_version)
    telemetry.send_event("optimum", "import", "import_from_optimum,ov_version:{}".format(get_rt_version()))
