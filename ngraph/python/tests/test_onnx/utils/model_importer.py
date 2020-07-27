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
import glob
import numpy as np
import os
import onnx
from onnx import numpy_helper, NodeProto, ModelProto
import onnx.backend.test
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase as OnnxTestCase
from onnx.backend.test.runner import TestItem
from typing import Any, Dict, List, Optional, Pattern, Set, Text, Type, Union
import unittest


class ModelImportRunner(onnx.backend.test.BackendTest):
    def __init__(
        self,
        backend: Type[Backend],
        models: List[Dict[str, str]],
        parent_module: Optional[str] = None
    ) -> None:
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns: Set[Pattern[Text]] = set()
        self._exclude_patterns: Set[Pattern[Text]] = set()
        self._test_items: Dict[Text, Dict[Text, TestItem]] = defaultdict(dict)

        for model in models:
            test_name = "test_{}".format(model["model_name"])

            test_case = OnnxTestCase(
                name=test_name,
                url=None,
                model_name=model["model_name"],
                model_dir=model["dir"],
                model=model["model_file"],
                data_sets=None,
                kind="OnnxBackendRealModelImportTest",
                rtol=model.get("rtol", 0.001),
                atol=model.get("atol", 1e-07),
            )
            self._add_model_test(test_case, "Validation")

    def _add_model_test(
        self,
        model_test: OnnxTestCase,
        kind: Text
    ) -> None:
        # model is loaded at runtime, note sometimes it could even
        # never loaded if the test skipped
        model_marker: List[Optional[Union[ModelProto, NodeProto]]] = [None]

        def run_import(test_self: Any, device: Text) -> None:
            if model_test.model_dir is None:
                raise unittest.SkipTest("Model directory not provided")
            else:
                model_dir = model_test.model_dir
            model_pb_path = os.path.join(model_dir, model_test.model)
            model = onnx.load(model_pb_path)
            model_marker[0] = model
            if not hasattr(self.backend, "is_compatible") and not callable(
                self.backend.is_compatible
            ):
                raise unittest.SkipTest("Provided backend does not provide is_compatible method")
            self.backend.is_compatible(model)

        def run_execution(test_self: Any, device: Text) -> None:
            if model_test.model_dir is None:
                raise unittest.SkipTest("Model directory not provided")
            else:
                model_dir = model_test.model_dir
            model_pb_path = os.path.join(model_dir, model_test.model)
            model = onnx.load(model_pb_path)

            prepared_model = self.backend.prepare(model, device)
            assert prepared_model is not None

            for test_data_npz in glob.glob(os.path.join(model_dir, "test_data_*.npz")):
                test_data = np.load(test_data_npz, encoding="bytes")
                inputs = list(test_data["inputs"])
                outputs = list(prepared_model.run(inputs))
                ref_outputs = test_data["outputs"]
                self.assert_similar_outputs(
                    ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
                )

            for test_data_dir in glob.glob(os.path.join(model_dir, "test_data_set*")):
                inputs = []
                inputs_num = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
                for i in range(inputs_num):
                    input_file = os.path.join(test_data_dir, "input_{}.pb".format(i))
                    tensor = onnx.TensorProto()
                    with open(input_file, "rb") as f:
                        tensor.ParseFromString(f.read())
                    inputs.append(numpy_helper.to_array(tensor))
                ref_outputs = []
                ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, "output_*.pb")))
                for i in range(ref_outputs_num):
                    output_file = os.path.join(test_data_dir, "output_{}.pb".format(i))
                    tensor = onnx.TensorProto()
                    with open(output_file, "rb") as f:
                        tensor.ParseFromString(f.read())
                    ref_outputs.append(numpy_helper.to_array(tensor))
                outputs = list(prepared_model.run(inputs))
                self.assert_similar_outputs(
                    ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
                )

        self._add_test(kind + "ModelImport", model_test.name, run_import, model_marker)
        self._add_test(kind + "ModelExecution", model_test.name, run_execution, model_marker)
