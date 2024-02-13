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
import contextlib
import io
import os
import pytest
import subprocess
import sys
import requests
import time
import zipfile


def download(test_data_dir, file_path):
    if file_path.exists():
        return file_path
    lock_path = test_data_dir / 'download.lock'
    with contextlib.suppress(FileNotFoundError, PermissionError):
        lock_path.unlink()
    for _ in range(9999):  # Give up after about 3 hours
        with contextlib.suppress(FileExistsError, PermissionError):
            with lock_path.open('bx'):
                if not file_path.exists():
                    response = requests.get("https://storage.openvinotoolkit.org/repositories/openvino/ci_dependencies/test/2021.4/samples_smoke_tests_data_2021.4.zip")
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zfile:
                        zfile.extractall(test_data_dir)
            lock_path.unlink(missing_ok=True)
            assert file_path.exists()
            return file_path
        time.sleep(1.0)


@pytest.mark.parametrize('counter', range(9999))
def test(counter, cache):
    test_data_dir = cache.mkdir('test_data')
    model = download(test_data_dir, test_data_dir / 'samples_smoke_tests_data_2021.4/models/public/squeezenet1.1/FP32/squeezenet1.1.xml')
    try:
        subprocess.check_output([sys.executable, '-c', 'import openvino as ov; core = ov.Core(); core.set_property({"ENABLE_MMAP": False}); core.read_model(r"' + f'{model}")'], stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=60.0)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(error.output)
        raise
