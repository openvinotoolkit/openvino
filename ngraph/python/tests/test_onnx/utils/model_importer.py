# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from collections import defaultdict
from typing import Dict, List, Optional, Pattern, Set, Text, Type

import onnx.backend.test
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase as OnnxTestCase
from onnx.backend.test.runner import TestItem


class ModelImportRunner(onnx.backend.test.BackendTest):
    def __init__(self, backend, models, parent_module=None):
        # type: (Type[Backend], List[Dict[str,str]], Optional[str]) -> None
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns = set()  # type: Set[Pattern[Text]]
        self._exclude_patterns = set()  # type: Set[Pattern[Text]]
        self._test_items = defaultdict(dict)  # type: Dict[Text, Dict[Text, TestItem]]

        for model in models:
            test_name = "test_{}".format(model["model_name"])

            test_case = OnnxTestCase(
                name=test_name,
                url=None,
                model_name=model["model_name"],
                model_dir=model["dir"],
                model=None,
                data_sets=None,
                kind="OnnxBackendRealModelImportTest",
                rtol=None,
                atol=None,
            )
            self._add_model_test(test_case, "Validation")

    def _add_model_test(self, model_test, kind):  # type: (TestCase, Text) -> None
        # model is loaded at runtime, note sometimes it could even
        # never loaded if the test skipped
        model_marker = [None]  # type: List[Optional[Union[ModelProto, NodeProto]]]

        def run(test_self, device):  # type: (Any, Text) -> None
            if model_test.model_dir is None:
                raise unittest.SkipTest('Model directory not provided')
            else:
                model_dir = model_test.model_dir
            model = onnx.load(model_dir)
            model_marker[0] = model
            if not hasattr(self.backend, 'is_compatible') \
               and not callable(self.backend.is_compatible):
                raise unittest.SkipTest(
                    'Provided backend does not provide is_compatible method')
            self.backend.is_compatible(model)

        self._add_test(kind + 'Model', model_test.name, run, model_marker)
