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
import logging as log
import os
import pytest
from common.samples_common_test_class import SamplesCommonTestClass
from common.samples_common_test_class import get_tests

test_data_fp32_async = get_tests \
    (cmd_params={'i': [os.path.join('227x227', 'dog.bmp')],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'api': ['async'],
                 'nireq': ['4'],
                 'niter': ['10'], },
     )

test_data_fp32_sync = get_tests \
    (cmd_params={'i': [os.path.join('227x227', 'dog.bmp')],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'niter': ['10'],
                 'api': ['sync']},
     )

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


class Test_benchmark_app(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'benchmark_app'
        super().setup_class()

    @pytest.mark.parametrize("param", test_data_fp32_async + test_data_fp32_sync)
    def test_data_fp32(self, param):
        stdout = self._test(param)
        assert "FPS" in stdout

    @pytest.mark.parametrize("param", test_data_fp32_async_config)
    def test_benchmark_app_sample_fp32_async_config(self, param):
        _check_output(self, param)

    @pytest.mark.parametrize("param", test_data_fp32_async_config_dump_nstreams)
    def test_benchmark_app_fp32_async_config_dump_nstreams(self, param):
        _check_output(self, param)
        _check_config(param)

    @pytest.mark.parametrize("param", test_data_fp16_config_dump_exec_graph)
    def test_benchmark_app_fp16_config_dump_auto(self, param):
        _check_output(self, param)
        _check_config(param)
        exec_graph_path = os.path.join(os.environ['WORKSPACE'], 'exec_graph.xml')
        assert os.path.exists(exec_graph_path) == True
        os.remove(exec_graph_path)
        assert os.path.exists(exec_graph_path) == False
        log.info('Execution graph check passed')

    @pytest.mark.parametrize("param", test_data_fp32_config_dump_pin)
    def test_benchmark_app_fp32_config_dump_pin(self, param):
        _check_output(self, param)
        _check_config(param)

    @pytest.mark.parametrize("param", test_data_fp32_reshape)
    def test_benchmark_app_test_data_fp32_reshape(self, param):
        _check_output(self, param)

    @pytest.mark.parametrize("param", test_data_dynamic_shapes_one_input)
    def test_benchmark_app_test_data_dynamic_shapes_diff_size(self, param):
        _check_output(self, param)


def stream_checker(param, config_json):
    device = param.get('d', 'CPU')
    if device not in config_json.keys():
        log.error("device not found")
        return False
    if 'NUM_STREAMS' not in config_json[device].keys():
        log.error("NUM_STREAMS not found")
        return False
    if param['nstreams'] == config_json[device]['NUM_STREAMS']:
        return True
    else:
        log.error("value of nstreams is false")
        return False

def pin_checker(param, config_json):
    device = param.get('d', 'CPU')
    if device not in config_json.keys():
        log.error("device not found")
        return False
    if 'AFFINITY' not in config_json[device].keys():
        log.error("AFFINITY not found")
        return False
    if param['pin'] == 'YES' and config_json[device]['AFFINITY'] == 'CORE':
        return True
    elif param['pin'] == 'NO' and config_json[device]['AFFINITY'] == 'NONE':
        return True
    elif param['pin'] == config_json[device]['AFFINITY']:
        return True
    else:
        log.error("value of pin is false")
        return False

def _check_output(self, param):
    """
    Benchmark_app has functional and accuracy testing.
    For accuracy the test checks if 'FPS' in output. If both exist - the est passed
    """

    # Run _test function, that returns stdout or 0.
    if 'dump_config' not in param:
        param['dump_config'] = os.path.join(os.environ['WORKSPACE'], 'config.json')
    stdout = self._test(param)
    print(stdout)
    config_file_name = param['dump_config']
    config = open(config_file_name, encoding='utf-8')
    lines = config.readlines()
    print('config file name:', param['dump_config'])
    for line in lines:
        print(line)
    config.seek(0, 0)
    if not config:
        return 0
    config_json = json.load(config)
    config.close()
    if not stdout:
        return 0
    stdout = stdout.split('\n')
    is_ok = False
    flag = 0
    for line in stdout:
        if 'FPS' in line:
            is_ok = True
    if is_ok == False:
        flag = 1
        log.error("No FPS in output")
    assert flag == 0, "Wrong output of FPS"

    is_ok = False
    if 'nstreams' in param:
        is_ok = stream_checker(param, config_json)
        if is_ok == False:
            log.error("No expected nstreams in output")
            assert False, "check nstreams failed"
    is_ok = False
    if 'pin' in param:
        is_ok = pin_checker(param, config_json)
        if is_ok == False:
            log.error("No expected pin in output")
            assert False, "check pin failed"

    log.info('Accuracy passed')


def _check_config(param):
    """
    Check whether config is stored and correct
    """
    config_path = os.path.join(os.environ['WORKSPACE'], 'config.json')
    assert os.path.exists(config_path) == True
    device = param['d']
    with open(config_path, encoding='utf-8') as config_file:
        config = json.load(config_file)
    if device in ['CPU', 'GPU']:
        assert device in config.keys()
        if param.get('api', 'async') == 'async':
            if 'nstreams' in param:
                assert config[device]['NUM_STREAMS'] == param['nstreams']
        else:
            assert config[device].get('NUM_STREAMS', 1) == 1

    os.remove(config_path)
