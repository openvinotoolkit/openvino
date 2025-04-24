# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import onnx.backend.test
import unittest
import dataclasses

from collections import defaultdict, namedtuple
from onnx import numpy_helper, NodeProto, ModelProto
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test.case.test_case import TestCase as OnnxTestCase
from onnx.backend.test.runner import TestItem
from pathlib import Path
from tests.tests_python.utils.onnx_helpers import import_onnx_model
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Text,
    Type,
    Union,
    Callable,
    Sequence,
)

# add post-processing function as part of test data
OnnxTestCase_fields = [field.name for field in dataclasses.fields(OnnxTestCase)]
ExtOnnxTestCase = dataclasses.make_dataclass(cls_name="TestCaseExt",
                                             fields=[*OnnxTestCase_fields, "post_processing"])


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
        self._xfail_patterns = set()  # type: Set[Pattern[Text]]

        strings = [str(data_root), ".onnx", "/", "\\", "-"]
        for model in models:
            test_name = f"test{model['model_name']}"
            for string in strings:
                if string in [str(data_root), ".onnx"]:
                    test_name = test_name.replace(string, "")
                else:
                    test_name = test_name.replace(string, "_")

            test_name = test_name.lower()

            test_case = ExtOnnxTestCase(
                name=test_name,
                url=None,
                model_name=model["model_name"],
                model_dir=model["dir"],
                model=model["model_file"],
                data_sets=None,
                kind="OnnxBackendRealModelTest",
                rtol=model.get("rtol", 0.001),
                atol=model.get("atol", 1e-07),
                __test__=True,
                post_processing=model.get("post_processing", None),
            )
            self._add_model_import_test(test_case)
            self._add_model_execution_test(test_case)

    @staticmethod
    def _load_onnx_model(model_dir: Path, filename: Path) -> ModelProto:
        if model_dir is None:
            raise unittest.SkipTest("Model directory not provided")

        return onnx.load(model_dir / filename)

    @classmethod
    def _execute_npz_data(
        cls,
        model_dir: str,
        prepared_model: BackendRep,
        result_rtol: float,
        result_atol: float,
        post_processing: Callable[[Sequence[Any]], Sequence[Any]] = None,
    ) -> int:
        executed_tests = 0
        for test_data_npz in model_dir.glob("test_data_*.npz"):
            test_data = np.load(test_data_npz, encoding="bytes")
            inputs = list(test_data["inputs"])
            outputs = list(prepared_model.run(inputs))
            ref_outputs = test_data["outputs"]
            if post_processing is not None:
                outputs = post_processing(outputs)
            cls.assert_similar_outputs(ref_outputs, outputs, result_rtol, result_atol)
            executed_tests = executed_tests + 1
        return executed_tests

    @classmethod
    def _execute_pb_data(
        cls,
        model_dir: str,
        prepared_model: BackendRep,
        result_rtol: float,
        result_atol: float,
        post_processing: Callable[[Sequence[Any]], Sequence[Any]] = None,
    ) -> int:
        executed_tests = 0
        for test_data_dir in model_dir.glob("test_data_set*"):
            inputs = []
            inputs_num = len(list(test_data_dir.glob("input_*.pb")))
            for i in range(inputs_num):
                input_file = Path(test_data_dir) / f"input_{i}.pb"
                tensor = onnx.TensorProto()
                with open(input_file, "rb") as f:
                    tensor.ParseFromString(f.read())
                inputs.append(numpy_helper.to_array(tensor))
            ref_outputs = []
            ref_outputs_num = len(list(test_data_dir.glob("output_*.pb")))
            for i in range(ref_outputs_num):
                output_file = Path(test_data_dir) / f"output_{i}.pb"
                tensor = onnx.TensorProto()
                with open(output_file, "rb") as f:
                    tensor.ParseFromString(f.read())
                ref_outputs.append(numpy_helper.to_array(tensor))
            if len(inputs) == 0:
                continue
            outputs = list(prepared_model.run(inputs))
            if post_processing is not None:
                outputs = post_processing(outputs)
            cls.assert_similar_outputs(ref_outputs, outputs, result_rtol, result_atol)
            executed_tests = executed_tests + 1
        return executed_tests


    def _add_model_import_test(self, model_test: ExtOnnxTestCase) -> None:
        # model is loaded at runtime, note sometimes it could even
        # never loaded if the test skipped
        model_marker = [None]  # type: List[Optional[Union[ModelProto, NodeProto]]]

        def run_import(test_self: Any, device: Text) -> None:
            model = ModelImportRunner._load_onnx_model(model_test.model_dir, model_test.model)
            model_marker[0] = model_test.model_dir / model_test.model
            assert import_onnx_model(model)

        self._add_test("ModelImport", model_test.name, run_import, model_marker)

    def _add_model_execution_test(self, model_test: ExtOnnxTestCase) -> None:
        # model is loaded at runtime, note sometimes it could even
        # never loaded if the test skipped
        model_marker = [None]  # type: List[Optional[Union[ModelProto, NodeProto]]]

        def run_execution(test_self: Any, device: Text) -> None:
            model = ModelImportRunner._load_onnx_model(model_test.model_dir, model_test.model)
            model_marker[0] = model_test.model_dir / model_test.model
            prepared_model = self.backend.prepare(model, device)
            assert prepared_model is not None
            executed_tests = ModelImportRunner._execute_npz_data(
                model_test.model_dir,
                prepared_model,
                model_test.rtol,
                model_test.atol,
                model_test.post_processing,
            )

            executed_tests = executed_tests + ModelImportRunner._execute_pb_data(
                model_test.model_dir,
                prepared_model,
                model_test.rtol,
                model_test.atol,
                model_test.post_processing,
            )
            assert executed_tests > 0, "This model has no test data"

        self._add_test("ModelExecution", model_test.name, run_execution, model_marker)
