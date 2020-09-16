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

import numpy as np
import onnx
import onnx.backend.test
import unittest

from collections import defaultdict
from onnx import numpy_helper, NodeProto, ModelProto
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test.case.test_case import TestCase as OnnxTestCase
from onnx.backend.test.runner import TestItem
from pathlib import Path
from tests.test_onnx.utils.onnx_helpers import import_onnx_model
from typing import Any, Dict, List, Optional, Pattern, Set, Text, Type, Union


class ModelImportRunner(onnx.backend.test.BackendTest):
    def __init__(
        self,
        backend: Type[Backend],
        models: List[Dict[str, Path]],
        parent_module: Optional[str] = None,
        data_root: Optional[Path] = "",
    ) -> None:
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns = set()  # type: Set[Pattern[Text]]
        self._exclude_patterns = set()  # type: Set[Pattern[Text]]
        self._test_items = defaultdict(dict)  # type: Dict[Text, Dict[Text, TestItem]]

        for model in models:
            test_name = "test{}".format(model["model_name"]) \
                .replace(str(data_root), "") \
                .replace(".onnx", "") \
                .replace("/", "_") \
                .replace("\\", "_") \
                .replace("-", "_")

            test_case = OnnxTestCase(
                name=test_name,
                url=None,
                model_name=model["model_name"],
                model_dir=model["dir"],
                model=model["model_file"],
                data_sets=None,
                kind="OnnxBackendRealModelTest",
                rtol=model.get("rtol", 0.001),
                atol=model.get("atol", 1e-07),
            )
            self._add_model_import_test(test_case)
            self._add_model_execution_test(test_case)

    @staticmethod
    def _load_onnx_model(model_dir: Path, filename: Path) -> ModelProto:
        if model_dir is None:
            raise unittest.SkipTest("Model directory not provided")

        return onnx.load(model_dir / filename)

    def _add_model_import_test(self, model_test: OnnxTestCase) -> None:
        # model is loaded at runtime, note sometimes it could even
        # never loaded if the test skipped
        model_marker = [None]  # type: List[Optional[Union[ModelProto, NodeProto]]]

        def run_import(test_self: Any, device: Text) -> None:
            model = ModelImportRunner._load_onnx_model(model_test.model_dir, model_test.model)
            model_marker[0] = model
            assert import_onnx_model(model)

        self._add_test("ModelImport", model_test.name, run_import, model_marker)

    @classmethod
    def _execute_npz_data(
        cls, model_dir: str, prepared_model: BackendRep, result_rtol: float, result_atol: float,
    ) -> int:
        executed_tests = 0
        for test_data_npz in model_dir.glob("test_data_*.npz"):
            test_data = np.load(test_data_npz, encoding="bytes")
            inputs = list(test_data["inputs"])
            outputs = list(prepared_model.run(inputs))
            ref_outputs = test_data["outputs"]
            cls.assert_similar_outputs(ref_outputs, outputs, result_rtol, result_atol)
            executed_tests = executed_tests + 1
        return executed_tests

    @classmethod
    def _execute_pb_data(
        cls, model_dir: str, prepared_model: BackendRep, result_rtol: float, result_atol: float,
    ) -> int:
        executed_tests = 0
        for test_data_dir in model_dir.glob("test_data_set*"):
            inputs = []
            inputs_num = len(list(test_data_dir.glob("input_*.pb")))
            for i in range(inputs_num):
                input_file = Path(test_data_dir) / "input_{}.pb".format(i)
                tensor = onnx.TensorProto()
                with open(input_file, "rb") as f:
                    tensor.ParseFromString(f.read())
                inputs.append(numpy_helper.to_array(tensor))
            ref_outputs = []
            ref_outputs_num = len(list(test_data_dir.glob("output_*.pb")))
            for i in range(ref_outputs_num):
                output_file = Path(test_data_dir) / "output_{}.pb".format(i)
                tensor = onnx.TensorProto()
                with open(output_file, "rb") as f:
                    tensor.ParseFromString(f.read())
                ref_outputs.append(numpy_helper.to_array(tensor))
            if(len(inputs) == 0):
                continue
            outputs = list(prepared_model.run(inputs))
            cls.assert_similar_outputs(ref_outputs, outputs, result_rtol, result_atol)
            executed_tests = executed_tests + 1
        return executed_tests

    def _add_model_execution_test(self, model_test: OnnxTestCase) -> None:
        # model is loaded at runtime, note sometimes it could even
        # never loaded if the test skipped
        model_marker = [None]  # type: List[Optional[Union[ModelProto, NodeProto]]]

        def run_execution(test_self: Any, device: Text) -> None:
            model = ModelImportRunner._load_onnx_model(model_test.model_dir, model_test.model)
            model_marker[0] = model
            prepared_model = self.backend.prepare(model, device)
            assert prepared_model is not None
            executed_tests = ModelImportRunner._execute_npz_data(
                model_test.model_dir, prepared_model, model_test.rtol, model_test.atol
            )

            executed_tests = executed_tests + ModelImportRunner._execute_pb_data(
                model_test.model_dir, prepared_model, model_test.rtol, model_test.atol
            )

            assert executed_tests > 0, "This model have no test data"
        self._add_test("ModelExecution", model_test.name, run_execution, model_marker)
