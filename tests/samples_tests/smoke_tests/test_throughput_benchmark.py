"""
 Copyright (C) 2018-2026 Intel Corporation
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
from pathlib import Path

import openvino as ov
from openvino import opset8 as opset

from common.samples_common_test_class import SamplesCommonTestClass, get_cmd_output_after_signal


class Test_throughput_benchmark_cpp(SamplesCommonTestClass):
    sample_name = 'throughput_benchmark'

    def test(self, cache):
        self._test({'m': 'bvlcalexnet-12.onnx'}, cache, use_preffix=False)


class Test_throughput_benchmark_py(SamplesCommonTestClass):
    sample_name = 'throughput_benchmark'
    executable_path = f'{sys.executable} -bb -W error -X dev -X warn_default_encoding "{os.environ["IE_APP_PYTHON_PATH"]}/benchmark/{sample_name}/{sample_name}.py"'

    def test(self, monkeypatch, cache):
        monkeypatch.setenv('PYTHONCOERCECLOCALE', 'warn')
        self._test({'m': 'bvlcalexnet-12.onnx'}, cache, use_preffix=False)

    def test_continuous_mode(self, tmp_path):
        model_path = tmp_path / 'model.xml'
        model_input = opset.parameter([1, 3, 32, 32], ov.Type.f32)

        ov.save_model(ov.Model([opset.relu(model_input)], [model_input]), model_path)

        sample_path = Path(os.environ['IE_APP_PYTHON_PATH']) / 'benchmark'/ self.sample_name / f'{self.sample_name}.py'
        output = get_cmd_output_after_signal(
            sys.executable, '-u', sample_path, model_path, '--seconds-to-run', '0',
            ready_message='Starting inference loop',
            output_path=tmp_path / 'throughput.log'
        )

        assert 'Count:' in output, output
        assert 'Throughput:' in output, output
