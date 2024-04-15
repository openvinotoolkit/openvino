# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.ovc.convert import convert_model
from openvino.tools.ovc.telemetry_utils import is_optimum, init_mo_telemetry

import importlib.metadata as importlib_metadata

try:
    optimum_version = importlib_metadata.version("optimum-intel")
except importlib_metadata.PackageNotFoundError:
    optimum_version = None

if is_optimum() and optimum_version is not None:
    telemetry = init_mo_telemetry("Optimum Intel", optimum_version)
    telemetry.send_event("ov", "import", "import_from_optimum")
else:
    telemetry = init_mo_telemetry()
    telemetry.send_event("ov", "import", "general_import")
