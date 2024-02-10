"""
 Copyright (C) 2018-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import io
import pytest
import requests
import zipfile


@pytest.fixture(scope='session', autouse=True)
def download_test_data(pytestconfig):
    test_data_dir = pytestconfig.cache.makedir('test_data_dir')
    one_of_the_required_files = test_data_dir / 'samples_smoke_tests_data_2021.4' / 'models' / 'public' / 'squeezenet1.1' / 'FP16' / 'squeezenet1.1.xml'
    if (one_of_the_required_files).exists():
        return
    response = requests.get("https://storage.openvinotoolkit.org/repositories/openvino/ci_dependencies/test/2021.4/samples_smoke_tests_data_2021.4.zip")
    if (one_of_the_required_files).exists():
        # Other process could have already extracted it
        return
    with zipfile.ZipFile(io.BytesIO(response.content)) as zfile:
        zfile.extractall(test_data_dir)
