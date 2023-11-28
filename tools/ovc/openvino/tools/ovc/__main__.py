# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from openvino.tools.ovc.main import main
from openvino.tools.ovc.telemetry_utils import init_mo_telemetry

init_mo_telemetry()
sys.exit(main())
