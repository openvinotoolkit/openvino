import sys

import openvino.tools.pot.api
import openvino.tools.pot.engines.ie_engine
import openvino.tools.pot.graph
import openvino.tools.pot.graph.model_utils
import openvino.tools.pot.pipeline.initializer


sys.modules["compression.api"] = openvino.tools.pot.api
sys.modules["compression.engines.ie_engine"] = openvino.tools.pot.engines.ie_engine
sys.modules["compression.graph"] = openvino.tools.pot.graph
sys.modules["compression.graph.model_utils"] = openvino.tools.pot.graph.model_utils
sys.modules["compression.pipeline.initializer"] = openvino.tools.pot.pipeline.initializer
