# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import warnings

import openvino.tools.pot.api
import openvino.tools.pot.engines
import openvino.tools.pot.graph
import openvino.tools.pot.pipeline


warnings.warn('Import compression is deprecated. Please use openvino.tools.pot instead', DeprecationWarning)

sys.modules["compression.api"] = openvino.tools.pot.api
sys.modules["compression.engines"] = openvino.tools.pot.engines
sys.modules["compression.engines.ie_engine"] = openvino.tools.pot.engines.ie_engine
sys.modules["compression.graph"] = openvino.tools.pot.graph
sys.modules["compression.graph.model_utils"] = openvino.tools.pot.graph.model_utils
sys.modules["compression.pipeline"] = openvino.tools.pot.pipeline
sys.modules["compression.pipeline.initializer"] = openvino.tools.pot.pipeline.initializer
