"""
 Copyright (C) 2024 Intel Corporation
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
import sys
from common.samples_common_test_class import SamplesCommonTestClass


class Test_sync_benchmark_cpp(SamplesCommonTestClass):
    sample_name = 'sync_benchmark'

    def test(self, cache):
        self._test({'m': 'bvlcalexnet-12.onnx'}, cache, use_preffix=False)


class Test_sync_benchmark_py(SamplesCommonTestClass):
    sample_name = 'sync_benchmark'
    executable_path = f'{sys.executable} -bb -W error -X dev -X warn_default_encoding "{os.environ["IE_APP_PYTHON_PATH"]}/benchmark/{sample_name}/{sample_name}.py"'

    def test(self, monkeypatch, cache):
        monkeypatch.setenv('PYTHONCOERCECLOCALE', 'warn')
        self._test({'m': 'bvlcalexnet-12.onnx'}, cache, use_preffix=False)
