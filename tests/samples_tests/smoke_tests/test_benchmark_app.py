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
import json
import os
import pathlib
import pytest
from common.samples_common_test_class import get_devices, get_cmd_output, prepend


def get_executable(sample_language):
    executable = 'benchmark_app'
    if sample_language == 'C++':
        executable = pathlib.Path(os.environ['IE_APP_PATH'], 'benchmark_app').with_suffix('.exe')
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
        with (tmp_path / 'conf.json').open() as file:
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
def test_benchmark_app(sample_language, api, nireq, device, cache):
    verify(sample_language, device, api=api, nireq=nireq, cache=cache)

test_data_fp32_async_config = get_tests \
    (cmd_params={'i': ['227x227/dog.bmp'],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'd': ['CPU'],
                 'api': ['async'],
                 'nireq': ['4'],
                 'niter': ['10'],
                 'nstreams' : ['2'],
                 'pin' : ['YES', 'NO', 'NUMA', 'HYBRID_AWARE'],
                 'dump_config' : [os.path.join(os.environ.get('WORKSPACE'), 'test_data_fp32_async_config.json')],
                 'hint' : ['none']},
     use_device=['d']
     )

test_data_fp32_async = get_tests \
    (cmd_params={'i': ['227x227/dog.bmp'],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'api': ['async'],
                 'nireq': ['4'],
                 'niter': ['10']},
     )

test_data_fp32_sync = get_tests \
    (cmd_params={'i': ['227x227/dog.bmp'],
                 'm':[os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'niter': ['10'],
                 'api': ['sync']},
     )

test_data_fp32_async_config_dump_nstreams = get_tests \
    (cmd_params={'i': ['227x227/dog.bmp'],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'd': ['CPU', 'GPU'],
                 'api': ['async'],
                 'nstreams' : ['2'],
                 'hint' : ['none'],
                 'niter': ['10'],
                 'dump_config' : [os.path.join(os.environ.get('WORKSPACE'), 'config.json')]},
     use_device=['d']
     )

test_data_fp16_config_dump_exec_graph = get_tests \
    (cmd_params={'i': ['227x227/dog.bmp'],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'api': ['sync', 'async'],
                 'niter': ['10'],
                 'exec_graph_path' : [os.path.join(os.environ.get('WORKSPACE'), 'exec_graph.xml')],
                 'dump_config' : [os.path.join(os.environ.get('WORKSPACE'), 'config.json')]},
     )

test_data_fp32_config_dump_pin = get_tests \
    (cmd_params={'i': ['227x227/dog.bmp'],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'd': ['CPU'],
                 'pin' : ['YES', 'NO', 'NUMA', 'HYBRID_AWARE'],
                 'hint': ['none'],
                 'niter': ['10'],
                 'dump_config' : [os.path.join(os.environ.get('WORKSPACE'), 'config.json')]},
     use_device=['d']
     )

test_data_fp32_reshape = get_tests \
    (cmd_params={'i': ['227x227/dog.bmp'],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'sample_type': ['C++', 'Python'],
                 'shape' : ['data[2,3,227,227]'],
                 'niter': ['10']},
     )

test_data_dynamic_shapes_one_input = get_tests \
    (cmd_params={'i': 2 * ['227x227/dog.bmp'],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'sample_type': ['C++'],
                 'shape' : ['[?,3,?,?]'],
                 'data_shape' : ['[1,3,227,227][1,3,227,227]'],
                 'layout' : ['[NCHW]'],
                 'niter': ['10']
                 },
     )


@pytest.mark.skipif('CPU' not in get_devices(), reason='affinity is a CPU property')
@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('pin', ['YES', 'NO', 'NUMA', 'HYBRID_AWARE'])
def test_affinity_setting(sample_language, pin, cache, tmp_path):
    verify(sample_language, 'CPU', pin=pin, nstreams='2', cache=cache, tmp_path=tmp_path)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', {'CPU', 'GPU'} & get_devices())
def test_affinity_setting_asfasdf(sample_language, device, cache, tmp_path):
    verify(sample_language, device, cache=cache, tmp_path=tmp_path)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('api', ['sync', 'async'])
@pytest.mark.parametrize('device', get_devices())
def test_affinity_setting_asfasdf_kek(sample_language, api, device, cache, tmp_path):
    verify(sample_language, device, api=api, cache=cache, tmp_path=tmp_path)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', get_devices())
def test_reshape(sample_language, device, cache):
    verify(sample_language, device, shape='data[2,3,227,227]', cache=cache)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', get_devices())
def test_dynamic_shape(sample_language, device, cache):
    verify(sample_language, device, shape='[?,3,?,?]', data_shape='[1,3,227,227][1,3,227,227]', layout='[NCHW]', cache=cache)
