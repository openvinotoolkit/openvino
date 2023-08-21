# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import openvino.tools.pot.api
import openvino.tools.pot.engines
import openvino.tools.pot.graph
import openvino.tools.pot.pipeline
from openvino.tools.pot.utils.logger import get_logger


logger = get_logger(__name__)
logger.warning('Import compression is deprecated. Please use openvino.tools.pot instead')


sys.modules["compression.api"] = openvino.tools.pot.api
sys.modules["compression.engines"] = openvino.tools.pot.engines
sys.modules["compression.engines.ie_engine"] = openvino.tools.pot.engines.ie_engine
sys.modules["compression.graph"] = openvino.tools.pot.graph
sys.modules["compression.graph.model_utils"] = openvino.tools.pot.graph.model_utils
sys.modules["compression.pipeline"] = openvino.tools.pot.pipeline
sys.modules["compression.pipeline.initializer"] = openvino.tools.pot.pipeline.initializer
