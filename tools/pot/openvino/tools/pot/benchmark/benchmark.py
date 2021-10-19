# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from tempfile import gettempdir
from importlib import import_module
from subprocess import check_output
import numpy as np
from openvino.inference_engine import IENetwork, IECore   # pylint: disable=E0611
from .infer_request_wrap import InferRequestsQueue
from ..graph.model_utils import save_model
from ..utils.logger import get_logger
from ..utils.utils import create_tmp_dir

logger = get_logger(__name__)
benchmark_cfg = {'nireq': 0, 'nstreams': None, 'nthreads': None, 'performance_count': False,
                 'batch_size': 0, 'niter': 100, 'duration_seconds': 30, 'api_type': 'async',
                 'cpu_bind_thread': 'YES', 'bench_test_number': 0, 'benchmark_cpp': False,
                 'benchmark_app_dir':""}
__MODEL_PATH__ = create_tmp_dir(gettempdir())

def set_benchmark_config(cfg):
    for key in cfg.keys():
        if key in benchmark_cfg.keys():
            benchmark_cfg[key] = cfg[key]
        else:
            logger.error('Illegal key {}'.format(key))

    benchmark_cfg['benchmark_cpp'] = False
    if 'benchmark_app_dir' in cfg and benchmark_cfg['benchmark_app_dir']:
        if os.path.exists(benchmark_cfg['benchmark_app_dir']):
            benchmark_cfg['benchmark_cpp'] = True
            return
        logger.warning("Fail to find benchmark_app in {}.".format(benchmark_cfg['benchmark_app_dir']))
    openvino_model = import_module(IENetwork.__module__)
    path = os.path.dirname(openvino_model.__file__)
    benchmark_cpp_app = path + "/../../../../../benchmark_app"
    if os.path.exists(benchmark_cpp_app):
        benchmark_cfg['benchmark_cpp'] = True
        benchmark_cfg['benchmark_app_dir'] = benchmark_cpp_app
        logger.info("Using benchamr_app in {}.".format(benchmark_cpp_app))
        return
    logger.info("Using python interface for benchmark.")

def benchmark_embedded(model=None, mf=None, api_type=None, duration_seconds=0, config=None):
    """ Perform benchmark with dummy inputs, return inference requests's latency' mesurement result.
        :param model: model to be benchmarked.
        :param mf: if model is not provided, model file name should be provided.
        :param api_type: sync or async.
        :param duration_seconds: durations in seconds.
        :param config: config dict to be changed.
        :return: latency metrics.
        """
    if config:
        set_benchmark_config(config)
    if duration_seconds != 0:
        benchmark_cfg['duration_seconds'] = duration_seconds
    if api_type:
        benchmark_cfg['api_type'] = api_type
    if model:
        model_name = 'tmp_benmark_model_{}'.format(benchmark_cfg['bench_test_number'])
        paths = save_model(model, __MODEL_PATH__.name, model_name)
        path_to_model_file = paths[0]['model']
    if mf:
        path_to_model_file = mf.strip()
        if '.xml' not in path_to_model_file:
            logger.error('{} is not an xml file.'.format(path_to_model_file))
            return None

    if not os.path.exists(path_to_model_file):
        logger.error('{} does not exist.'.format(path_to_model_file))
        return None
    if benchmark_cfg['benchmark_cpp']:
        return benchmark_embedded_cpp_app(path_to_model_file)
    return benchmark_embedded_python_api(path_to_model_file)


def benchmark_embedded_python_api(path_to_model_file):
    """ Perform benchmark with dummy inputs, return inference requests's latency' mesurement result.
        :param path_to_model_file: if model is not provided, xml model file name.
        :return: latency metrics.
        """
    def get_dummy_inputs(batch_size, input_info, requests):
        """ Generate dummpy inputs based on input and batch information.
            :param batch_size: batch size
            :param input_info: network's input infor
            :param requests: the network's requests
            :return: requests_input_data
            """
        requests_input_data = []
        input_data = {}
        np_d_type = {'FP64': np.float64, 'I32': np.int32, 'FP32': np.float32, 'FP16': np.float16,
                     'U16': np.uint16, 'I16': np.int16, 'U8': np.uint8, 'I8': np.int8}
        for key, value in input_info.items():
            m = []
            dt = np_d_type[value.precision]
            for x in value.shape:
                m.append(x)
            m[0] = m[0] * batch_size
            input_data[key] = np.empty(tuple(m), dtype=dt)
        for _ in range(len(requests)):
            requests_input_data.append(input_data)
        return requests_input_data

    xml_filename = path_to_model_file
    bin_filename = path_to_model_file[:(len(path_to_model_file) - 4)] + '.bin'
    if not os.path.exists(bin_filename):
        logger.error('{} does not exist.'.format(bin_filename))
        return None

    ie = IECore()
    ie_network = ie.read_network(xml_filename, bin_filename)
    device = 'CPU'
    config = {'PERF_COUNT': 'NO'}
    ie.set_config({'CPU_BIND_THREAD': str(benchmark_cfg['cpu_bind_thread'])}, device)
    if benchmark_cfg['nthreads'] is not None and benchmark_cfg['nthreads']:
        ie.set_config({'CPU_THREADS_NUM': str(benchmark_cfg['nthreads'])}, device)
    if benchmark_cfg['nstreams'] is not None:
        ie.set_config({'CPU_THROUGHPUT_STREAMS': str(benchmark_cfg['nstreams'])}, device)
    exe_network = ie.load_network(ie_network, device, config=config, num_requests=benchmark_cfg['nireq'])
    infer_requests = exe_network.requests
    batch_size = ie_network.batch_size
    request_queue = InferRequestsQueue(infer_requests)
    requests_input_data = get_dummy_inputs(batch_size, ie_network.inputs, infer_requests)
    infer_request = request_queue.get_idle_request()

    # For warming up
    if benchmark_cfg['api_type'] == 'sync':
        infer_request.infer(requests_input_data[infer_request.id])
    else:
        infer_request.start_async(requests_input_data[infer_request.id])

    request_queue.wait_all()
    request_queue.reset_times()
    start_time = datetime.now()
    exec_time = (datetime.now() - start_time).total_seconds()
    iteration = 0
    logger.info('Starting benchmark, will be done in {} seconds with {} api via python interface.'
                .format(benchmark_cfg['duration_seconds'], benchmark_cfg['api_type']))

    while exec_time < benchmark_cfg['duration_seconds']:
        infer_request = request_queue.get_idle_request()
        if not infer_request:
            raise Exception('No idle Infer Requests!')
        if benchmark_cfg['api_type'] == 'sync':
            infer_request.infer(requests_input_data[infer_request.id])
        else:
            infer_request.start_async(requests_input_data[infer_request.id])
        iteration += 1
        exec_time = (datetime.now() - start_time).total_seconds()

    request_queue.wait_all()
    t = np.array(request_queue.times)
    q75, q25 = np.percentile(t, [75, 25])
    IQR = q75 - q25
    filtered_times = t[t < (q75 + 1.5 * IQR)]
    logger.debug('benchmark result: latency_filtered_mean:{0:.3f}ms, latency_minum: {1:.3f}ms, \
    using {2} requests of total {3} ones for latency calcluation.'.format(
        filtered_times.mean(), filtered_times.min(), filtered_times.size, t.size))
    del exe_network
    del ie
    del ie_network
    return filtered_times.mean()

def benchmark_embedded_cpp_app(path_to_model_file):
    """ Perform benchmark with dummy inputs, return inference requests's latency' mesurement result.
        :param path_to_model_file: xml model file name
        :return: latency metrics, and performance counts for each layer if layer_awareness is Ture.
        """

    cmd_cb = [benchmark_cfg["benchmark_app_dir"], "-api", benchmark_cfg["api_type"], "-m",
              path_to_model_file]
    if benchmark_cfg["duration_seconds"] != 0:
        cmd_cb.append("-t")
        cmd_cb.append(str(benchmark_cfg["duration_seconds"]))
    if benchmark_cfg['nthreads'] is not None:
        cmd_cb.append("-nthreads")
        cmd_cb.append(str(benchmark_cfg['nthreads']))
    if benchmark_cfg['batch_size'] != 0:
        cmd_cb.append("-b")
        cmd_cb.append(str(benchmark_cfg['batch_size']))
    if benchmark_cfg['nstreams'] is not None:
        cmd_cb.append("-nstreams")
        cmd_cb.append(str(benchmark_cfg['nstreams']))
    cmd_cb.append("-pin")
    cmd_cb.append(str(benchmark_cfg['cpu_bind_thread']))
    if benchmark_cfg['nireq'] != 0:
        cmd_cb.append("-nireq")
        cmd_cb.append(str(benchmark_cfg['nireq']))

    benchmark_finished = False
    logger.info(" ".join(cmd_cb))
    output = check_output(cmd_cb, shell=False)
    out = output.decode().split("\n")
    for line in out:
        if "Latency:" in line:
            latency_report = line.split()
            latency = float(latency_report[1])
            benchmark_finished = True
        if "Throughput:" in line:
            throughput_report = line.split()
            throughput = float(throughput_report[1])
    if not benchmark_finished:
        logger.error("Benchmark running fails.")
    logger.info("Benchmark result: latency: {0:.3f} ms, throughput: {1:.2f} FPS.".format(latency, throughput))
    return latency
