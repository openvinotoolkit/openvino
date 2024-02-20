"""
 Copyright (C) 2018-2024 Intel Corporation
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
import json
import os
import pathlib
import pytest
from common.samples_common_test_class import get_devices, get_cmd_output, prepend


def get_executable(sample_language):
    executable = 'benchmark_app'
    if sample_language == 'C++':
        executable = pathlib.Path(os.environ['IE_APP_PATH'], 'benchmark_app').with_suffix('.exe' if os.name == 'nt' else '')
        assert executable.exists()
        return executable
    return 'benchmark_app'


def verify(sample_language, device, api=None, nireq=None, shape=None, data_shape=None, nstreams=None, layout=None, pin=None, cache=None, tmp_path=None):
    output = get_cmd_output(
        get_executable(sample_language),
        *prepend(cache, '227x227/dog.bmp', 'squeezenet1.1/FP32/squeezenet1.1.xml'),
        *('-nstreams', nstreams) if nstreams else '',
        *('-layout', layout) if layout else '',
        *('-nireq', nireq) if nireq else '',
        *('-shape', shape) if shape else '',
        *('-data_shape', data_shape) if data_shape else '',
        *('-hint', 'none') if nstreams or pin else '',
        *('-pin', pin) if pin else '',
        *('-api', api) if api else '',
        *('-dump_config', tmp_path / 'conf.json') if tmp_path else '',
        *('-exec_graph_path', tmp_path / 'exec_graph.xml') if tmp_path else '',
        '-d', device, '-b', '1', '-niter', '10'
    )
    assert 'FPS' in output
    if tmp_path:
        assert (tmp_path / 'exec_graph.xml').exists()
        with (tmp_path / 'conf.json').open(encoding='utf-8') as file:
            config_json = json.load(file)
        if 'CPU' == device:
            assert 'CPU' in config_json
            assert not nstreams or config_json['CPU']['NUM_STREAMS'] == nstreams
            assert (not pin
                or pin == 'YES' and config_json['CPU']['AFFINITY'] == 'CORE'
                or pin == 'NO' and config_json['CPU']['AFFINITY'] == 'NONE'
                or pin == config_json['CPU']['AFFINITY'])


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
def test_benchmark_app_help(sample_language):
    get_cmd_output(get_executable(sample_language), '-h')


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('api', ['sync', 'async'])
@pytest.mark.parametrize('nireq', ['4', ''])
@pytest.mark.parametrize('device', get_devices())
def test_nireq(sample_language, api, nireq, device, cache):
    verify(sample_language, device, api=api, nireq=nireq, cache=cache)


@pytest.mark.skipif('CPU' not in get_devices(), reason='affinity is a CPU property')
@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('pin', ['YES', 'NO', 'NUMA', 'HYBRID_AWARE'])
def test_pin(sample_language, pin, cache, tmp_path):
    verify(sample_language, 'CPU', pin=pin, nstreams='2', cache=cache, tmp_path=tmp_path)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', sorted({'CPU', 'GPU'} & set(get_devices())))  # Determenisitic order is required for --numprocesses
def test_simple(sample_language, device, cache, tmp_path):
    verify(sample_language, device, cache=cache, tmp_path=tmp_path)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('api', ['sync', 'async'])
@pytest.mark.parametrize('device', get_devices())
def test_api(sample_language, api, device, cache, tmp_path):
    verify(sample_language, device, api=api, cache=cache, tmp_path=tmp_path)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', get_devices())
def test_reshape(sample_language, device, cache):
    verify(sample_language, device, shape='data[2,3,227,227]', cache=cache)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', get_devices())
def test_dynamic_shape(sample_language, device, cache):
    verify(sample_language, device, shape='[?,3,?,?]', data_shape='[1,3,227,227][1,3,227,227]', layout='[NCHW]', cache=cache)
