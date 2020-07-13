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

import glob
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Pattern, Set, Text, Type

import onnx.backend.test
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase as OnnxTestCase
from onnx.backend.test.runner import TestItem
from retrying import retry
from six.moves.urllib.request import urlopen, urlretrieve


class ModelZooTestRunner(onnx.backend.test.BackendTest):
    def __init__(self, backend, zoo_models, parent_module=None):
        # type: (Type[Backend], List[Dict[str,str]], Optional[str]) -> None
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns = set()  # type: Set[Pattern[Text]]
        self._exclude_patterns = set()  # type: Set[Pattern[Text]]
        self._test_items = defaultdict(dict)  # type: Dict[Text, Dict[Text, TestItem]]

        for zoo_model in zoo_models:
            test_name = "test_{}".format(zoo_model["model_name"])

            test_case = OnnxTestCase(
                name=test_name,
                url=zoo_model["url"],
                model_name=zoo_model["model_name"],
                model_dir=None,
                model=None,
                data_sets=None,
                kind="OnnxBackendRealModelTest",
                rtol=zoo_model.get("rtol", 0.001),
                atol=zoo_model.get("atol", 1e-07),
            )
            self._add_model_test(test_case, "Zoo")

    @staticmethod
    @retry
    def _get_etag_for_url(url):  # type: (str) -> str
        request = urlopen(url)
        return request.info().get("ETag")

    @staticmethod
    def _read_etag_file(model_dir):  # type: (str) -> str
        etag_file_path = os.path.join(model_dir, "source_tar_etag")
        if os.path.exists(etag_file_path):
            return open(etag_file_path).read()

    @staticmethod
    def _write_etag_file(model_dir, etag_value):  # type: (str, str) -> None
        etag_file_path = os.path.join(model_dir, "source_tar_etag")
        open(etag_file_path, "w").write(etag_value)

    @staticmethod
    def _backup_old_version(model_dir):  # type: (str) -> None
        if os.path.exists(model_dir):
            backup_index = 0
            while True:
                dest = "{}.old.{}".format(model_dir, backup_index)
                if os.path.exists(dest):
                    backup_index += 1
                    continue
                shutil.move(model_dir, dest)
                break

    @classmethod
    @retry
    def prepare_model_data(cls, model_test):  # type: (OnnxTestCase) -> Text
        onnx_home = os.path.expanduser(os.getenv("ONNX_HOME", os.path.join("~", ".onnx")))
        models_dir = os.getenv("ONNX_MODELS", os.path.join(onnx_home, "models"))
        model_dir = os.path.join(models_dir, model_test.model_name)  # type: Text
        current_version_etag = ModelZooTestRunner._get_etag_for_url(model_test.url)

        # If model already exists, check if it's the latest version by verifying cached Etag value
        if os.path.exists(os.path.join(model_dir, "model.onnx")):
            if not current_version_etag or current_version_etag == ModelZooTestRunner._read_etag_file(
                model_dir
            ):
                return model_dir

            # If model does exist, but is not current, backup directory
            ModelZooTestRunner._backup_old_version(model_dir)

        # Download and extract model and data
        download_file = tempfile.NamedTemporaryFile(delete=False)
        temp_clean_dir = tempfile.mkdtemp()

        try:
            download_file.close()
            print("\nStart downloading model {} from {}".format(model_test.model_name, model_test.url))
            urlretrieve(model_test.url, download_file.name)
            print("Done")

            with tempfile.TemporaryDirectory() as temp_extract_dir:
                with tarfile.open(download_file.name) as tar_file:
                    tar_file.extractall(temp_extract_dir)

                # Move model `.onnx` file from temp_extract_dir to temp_clean_dir
                model_files = glob.glob(temp_extract_dir + "/**/*.onnx", recursive=True)
                assert len(model_files) > 0, "Model file not found for {}".format(model_test.name)
                model_file = model_files[0]
                shutil.move(model_file, temp_clean_dir + "/model.onnx")

                # Move extracted test data sets to temp_clean_dir
                test_data_sets = glob.glob(temp_extract_dir + "/**/test_data_set_*", recursive=True)
                test_data_sets.extend(glob.glob(temp_extract_dir + "/**/test_data_*.npz", recursive=True))
                for test_data_set in test_data_sets:
                    shutil.move(test_data_set, temp_clean_dir)

                # Save Etag value to Etag file
                ModelZooTestRunner._write_etag_file(temp_clean_dir, current_version_etag)

                # Move temp_clean_dir to ultimate destination
                shutil.move(temp_clean_dir, model_dir)

        except Exception as e:
            print("Failed to prepare data for model {}: {}".format(model_test.model_name, e))
            os.remove(temp_clean_dir)
            raise
        finally:
            os.remove(download_file.name)
        return model_dir
