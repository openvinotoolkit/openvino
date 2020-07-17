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

MODELS_ROOT_DIR = tests.ADDITIONAL_MODELS_DIR
if (len(MODELS_ROOT_DIR) > 0):
    zoo_models = []
    for path in Path(MODELS_ROOT_DIR).rglob('*.onnx'):
        mdir, file = os.path.split(path)
        # Since directory structure is not uniform just grab last two directories
        # and join them with model file name without .onnx extension.
        name = '_'.join(mdir.split(os.path.sep)[-2:]) + '_' + file[:file.rfind('.')]

        zoo_models.append({
            'model_name': name,
            'dir': str(path)
        })

    sorted(zoo_models, key=itemgetter('model_name'))

    # Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
    OpenVinoOnnxBackend.backend_name = tests.BACKEND_NAME

    # import all test cases at global scope to make them visible to pytest
    backend_test = ModelImportRunner(OpenVinoOnnxBackend, zoo_models, __name__)
    test_cases = backend_test.test_cases["OnnxBackendValidationModelTest"]

    globals().update(backend_test.enable_report().test_cases)
