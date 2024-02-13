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
import subprocess
import sys
import requests
import zipfile
import multiprocessing
import pathlib


def verify_single_process(model):
    import openvino as ov
    core = ov.Core()
    core.set_property({"ENABLE_MMAP": False})
    core.read_model(model)


def download(test_data_dir, file_path):
    if not file_path.exists():
        response = requests.get("https://storage.openvinotoolkit.org/repositories/openvino/ci_dependencies/test/2021.4/samples_smoke_tests_data_2021.4.zip")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zfile:
            zfile.extractall(test_data_dir)
    verify_single_process(file_path)
    return file_path


def starter(model):
    try:
        subprocess.check_output([sys.executable, '-c', 'import openvino as ov; core = ov.Core(); core.set_property({"ENABLE_MMAP": False}); core.read_model(r"' + f'{model}");'], stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8', timeout=60.0)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(error.output)
        raise


def main():
    test_data_dir = pathlib.Path('.')
    model = download(test_data_dir, test_data_dir / 'samples_smoke_tests_data_2021.4/models/public/squeezenet1.1/FP32/squeezenet1.1.xml')
    pool = multiprocessing.Pool(processes=16)
    pool.map(starter, [model] * 999)


if __name__ == '__main__':
    main()
