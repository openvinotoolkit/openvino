"""
 Copyright (C) 2018-2025 Intel Corporation
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
import platform
import numpy as np
import pathlib
import pytest
from common.samples_common_test_class import get_devices, get_cmd_output, prepend
from openvino.runtime import opset8 as opset
import openvino.runtime as ov

def get_executable(sample_language):
    executable = 'benchmark_app'
    if sample_language == 'C++':
        executable = pathlib.Path(os.environ['IE_APP_PATH'], 'benchmark_app').with_suffix('.exe' if os.name == 'nt' else '')
        assert executable.exists()
        return executable
    return 'benchmark_app'

def create_random_4bit_bin_file(tmp_path, shape, name):
    fullname = tmp_path / name
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    pack_shape = [x for x in shape]
    pack_shape[-1] = pack_shape[-1]*4
    rand_data = (rs.uniform(0, 15, list(pack_shape)) >= 7).astype(int).flatten()
    raw_data = np.packbits(rand_data)
    with open(fullname, "wb") as f:
        f.write(raw_data)


def verify(sample_language, device, api=None, nireq=None, shape=None, data_shape=None, nstreams=None,
           layout=None, pin=None, cache=None, tmp_path=None, model='bvlcalexnet-12.onnx',
           inp='dog-224x224.bmp', batch='1', niter='10', max_irate=None, tm=None):
    output = get_cmd_output(
        get_executable(sample_language),
        *prepend(cache, inp, model, tmp_path),
        *('-nstreams', nstreams) if nstreams else '',
        *('-layout', layout) if layout else '',
        *('-nireq', nireq) if nireq else '',
        *('-max_irate', max_irate) if max_irate else '',
        *('-shape', shape) if shape else '',
        *('-data_shape', data_shape) if data_shape else '',
        *('-hint', 'none') if nstreams or pin else '',
        *('-pin', pin) if pin else '',
        *('-api', api) if api else '',
        *('-dump_config', tmp_path / 'conf.json') if tmp_path else '',
        *('-exec_graph_path', tmp_path / 'exec_graph.xml') if tmp_path else '',
        *('-b', batch) if batch else '',
        *('-niter', niter) if niter else '10',
        *('-t', tm) if tm else '',
        '-d', device
    )
    assert 'FPS' in output

    # No Windows support due to the lack of the ‘psutil’ module in the CI infrastructure
    # No Macos support due to no /proc/self/status file
    if platform.system() == "Linux":
        assert 'Compile model ram used' in output

    if tmp_path:
        assert (tmp_path / 'exec_graph.xml').exists()
        with (tmp_path / 'conf.json').open(encoding='utf-8') as file:
            config_json = json.load(file)
        if 'CPU' == device:
            assert 'CPU' in config_json
            assert not nstreams or config_json['CPU']['NUM_STREAMS'] == nstreams
            assert (not pin
                or pin == 'YES' and config_json['CPU']['ENABLE_CPU_PINNING'] == 'YES'
                or pin == 'NO' and config_json['CPU']['ENABLE_CPU_PINNING'] == 'NO')


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
def test_benchmark_app_help(sample_language):
    get_cmd_output(get_executable(sample_language), '-h')


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('api', ['sync', 'async'])
@pytest.mark.parametrize('nireq', ['4', ''])
@pytest.mark.parametrize('device', get_devices())
def test_nireq(sample_language, api, nireq, device, cache, tmp_path):
    verify(sample_language, device, api=api, nireq=nireq, cache=cache, tmp_path=tmp_path)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('max_irate', ['', '0', '10'])
@pytest.mark.parametrize('device', get_devices())
def test_max_irate(sample_language, device, max_irate, cache, tmp_path):
    verify(sample_language, device, max_irate=max_irate, cache=cache, tmp_path=tmp_path)


@pytest.mark.skipif('CPU' not in get_devices(), reason='affinity is a CPU property')
@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('pin', ['YES', 'NO'])
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
def test_reshape(sample_language, device, cache, tmp_path):
    verify(sample_language, device, shape='data_0[2,3,224,224]', cache=cache, tmp_path=tmp_path)


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', get_devices())
def test_dynamic_shape(sample_language, device, cache, tmp_path):
    verify(sample_language, device, model='efficientnet-lite4-11-qdq.onnx',
           shape='[?,224,224,3]', data_shape='[1,224,224,3][2,224,224,3]', layout='[NHWC]', cache=cache, tmp_path=tmp_path)

@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', ['CPU'])
@pytest.mark.parametrize('inp', [None, 'random_4bit_data.bin'])
def test_4bit_precision_input(sample_language, device, inp, cache, tmp_path):
    inp_type = ov.Type.i4
    inp_shape = [128] # only pass scalar.
    input = opset.parameter(inp_shape, inp_type, name='in')
    cvt = opset.convert(input, ov.Type.f32)
    result = opset.result(cvt, name='cvt')
    model_4bit = ov.Model([result], [input], 'model_with_4bit_input')
    if inp != None and inp.endswith(".bin"):
        create_random_4bit_bin_file(tmp_path, inp_shape, inp)
    verify(sample_language, device, model=model_4bit, inp=inp, cache=cache, tmp_path=tmp_path, batch=None, tm='1')


@pytest.mark.parametrize('sample_language', ['C++', 'Python'])
@pytest.mark.parametrize('device', get_devices())
def test_out_of_tensor_size_range_npy_multibatch(sample_language, device, cache, tmp_path):
    inp = tmp_path / 'batch2.npy'
    with open(inp, "wb") as batch2_npy:
        np.save(
            batch2_npy,
            np.random.RandomState(
                np.random.MT19937(np.random.SeedSequence(0))
            ).uniform(0, 256, [2, 3, 224, 224]).astype(np.uint8)
        )
    # benchmark_app reads batch from model or cmd, not from npy.
    # benchmark_app still verifyes npy shape for python impl.
    verify(sample_language, device, inp=inp, cache=cache, tmp_path=tmp_path, batch='2')
