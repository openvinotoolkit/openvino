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

import tests
from operator import itemgetter
from pathlib import Path
import os

from tests.test_onnx.utils import OpenVinoOnnxBackend
from tests.test_onnx.utils.model_importer import ModelImportRunner


def _get_default_additional_models_dir():
    onnx_home = os.path.expanduser(os.getenv("ONNX_HOME", os.path.join("~", ".onnx")))
    return os.path.join(onnx_home, "additional_models")


MODELS_ROOT_DIR = tests.ADDITIONAL_MODELS_DIR
if len(MODELS_ROOT_DIR) == 0:
    MODELS_ROOT_DIR = _get_default_additional_models_dir()

zoo_models = []
# rglob doesn't work for symlinks, so models have to be physically somwhere inside "MODELS_ROOT_DIR"
for path in Path(MODELS_ROOT_DIR).rglob("*.onnx"):
    mdir, file = os.path.split(str(path))
    if not file.startswith("."):
        zoo_models.append({"model_name": path, "model_file": file, "dir": str(mdir)})

if len(zoo_models) > 0:
    sorted(zoo_models, key=itemgetter("model_name"))

    # Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
    OpenVinoOnnxBackend.backend_name = tests.BACKEND_NAME

    # import all test cases at global scope to make them visible to pytest
    backend_test = ModelImportRunner(OpenVinoOnnxBackend, zoo_models, __name__)
    test_cases = backend_test.test_cases["OnnxBackendValidationModelImportTest"]
    del test_cases

    test_cases = backend_test.test_cases["OnnxBackendValidationModelExecutionTest"]
    del test_cases

    globals().update(backend_test.enable_report().test_cases)
