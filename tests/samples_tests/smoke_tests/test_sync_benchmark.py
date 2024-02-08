"""
 Copyright (C) 2018-2024 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import pathlib
import sys
from common.samples_common_test_class import SamplesCommonTestClass


class TestSyncBenchmarkCpp(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'sync_benchmark'
        super().setup_class()

    def test_benchmark_app_onnx(self):
        self._test({'m': str(pathlib.PurePath('squeezenet_v1.1') / 'FP32' / 'squeezenet1.1.xml')}, use_preffix=False)


class TestSyncBenchmarkPython(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'sync_benchmark'
        cls.executable_path = sys.executable + ' -bb -W error -X dev -X warn_default_encoding ' + str((pathlib.PurePath(os.environ['IE_APP_PYTHON_PATH']) / 'benchmark' / cls.sample_name / cls.sample_name).with_suffix('.py'))
        super().setup_class()

    def test_benchmark_app_onnx(self, monkeypatch):
        monkeypatch.setenv('PYTHONCOERCECLOCALE', 'warn')
        self._test({'m': str(pathlib.PurePath('squeezenet_v1.1') / 'FP32' / 'squeezenet1.1.xml')}, use_preffix=False)
