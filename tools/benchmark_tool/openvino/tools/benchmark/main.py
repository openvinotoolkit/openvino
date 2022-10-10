# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from datetime import datetime

from openvino.runtime import Dimension

from openvino.tools.benchmark.benchmark import Benchmark
from openvino.tools.benchmark.parameters import parse_args
from openvino.tools.benchmark.utils.constants import MULTI_DEVICE_NAME, HETERO_DEVICE_NAME, CPU_DEVICE_NAME, \
    GPU_DEVICE_NAME, MYRIAD_DEVICE_NAME, GNA_DEVICE_NAME, BLOB_EXTENSION
from openvino.tools.benchmark.utils.inputs_filling import get_input_data
from openvino.tools.benchmark.utils.logging import logger
from openvino.tools.benchmark.utils.progress_bar import ProgressBar
from openvino.tools.benchmark.utils.utils import next_step, get_number_iterations, pre_post_processing, \
    process_help_inference_string, print_perf_counters, dump_exec_graph, get_duration_in_milliseconds, \
    get_command_line_arguments, parse_value_per_device, parse_devices, get_inputs_info, \
    print_inputs_and_outputs_info, get_network_batch_size, load_config, dump_config, get_latency_groups, \
    check_for_static, can_measure_as_static
from openvino.tools.benchmark.utils.statistics_report import StatisticsReport, averageCntReport, detailedCntReport

def parse_and_check_command_line():
    def arg_not_empty(arg_value,empty_value):
        return not arg_value is None and not arg_value == empty_value

    args = parse_args()

    if not args.perf_hint == "none" and (arg_not_empty(args.number_streams, "") or arg_not_empty(args.number_threads, 0) or arg_not_empty(args.infer_threads_pinning, "")):
        raise Exception("-nstreams, -nthreads and -pin options are fine tune options. To use them you " \
                        "should explicitely set -hint option to none. This is not OpenVINO limitation " \
                        "(those options can be used in OpenVINO together), but a benchmark_app UI rule.")
    
    if args.report_type == "average_counters" and "MULTI" in args.target_device:
        raise Exception("only detailed_counters report type is supported for MULTI device")
    
    _, ext = os.path.splitext(args.path_to_model)
    is_network_compiled = True if ext == BLOB_EXTENSION else False
    is_precisiton_set = not (args.input_precision == "" and args.output_precision == "" and args.input_output_precision == "")

    if is_network_compiled and is_precisiton_set:
        raise Exception("Cannot set precision for a compiled network. " \
                        "Please re-compile your network with required precision " \
                        "using compile_tool")
    
    return args, is_network_compiled

def main():
    statistics = None
    try:
        # ------------------------------ 1. Parsing and validating input arguments ------------------------------
        next_step()
        args, is_network_compiled = parse_and_check_command_line()

        if args.number_streams is None:
                logger.warning(" -nstreams default value is determined automatically for a device. "
                               "Although the automatic selection usually provides a reasonable performance, "
                               "but it still may be non-optimal for some cases, for more information look at README. ")

        command_line_arguments = get_command_line_arguments(sys.argv)
        if args.report_type:
          statistics = StatisticsReport(StatisticsReport.Config(args.report_type, args.report_folder))
          statistics.add_parameters(StatisticsReport.Category.COMMAND_LINE_PARAMETERS, command_line_arguments)

        def is_flag_set_in_command_line(flag):
            return any(x.strip('-') == flag for x, y in command_line_arguments)

        device_name = args.target_device

        devices = parse_devices(device_name)
        device_number_streams = parse_value_per_device(devices, args.number_streams, "nstreams")
        device_infer_precision = parse_value_per_device(devices, args.infer_precision, "infer_precision")

        config = {}
        if args.load_config:
            load_config(args.load_config, config)

        if is_network_compiled:
            print("Model is compiled")

        # ------------------------------ 2. Loading OpenVINO ---------------------------------------------------
        next_step(step_id=2)

        benchmark = Benchmark(args.target_device, args.number_infer_requests,
                              args.number_iterations, args.time, args.api_type, args.inference_only)

        ## CPU (OneDNN) extensions
        if CPU_DEVICE_NAME in device_name and args.path_to_extension:
            benchmark.add_extension(path_to_extension=args.path_to_extension)

        ## GPU (clDNN) Extensions
        if GPU_DEVICE_NAME in device_name and args.path_to_cldnn_config:
            if GPU_DEVICE_NAME not in config.keys():
                config[GPU_DEVICE_NAME] = {}
            config[GPU_DEVICE_NAME]['CONFIG_FILE'] = args.path_to_cldnn_config

        if GPU_DEVICE_NAME in config.keys() and 'CONFIG_FILE' in config[GPU_DEVICE_NAME].keys():
            cldnn_config = config[GPU_DEVICE_NAME]['CONFIG_FILE']
            benchmark.add_extension(path_to_cldnn_config=cldnn_config)

        for device in devices:
            supported_properties = benchmark.core.get_property(device, 'SUPPORTED_PROPERTIES')
            if 'PERFORMANCE_HINT' in supported_properties:
                if is_flag_set_in_command_line('hint'):
                    if args.perf_hint=='none':
                        logger.warning(f"No device {device} performance hint is set.")
                        args.perf_hint = ''
                else:
                    args.perf_hint = "THROUGHPUT" if benchmark.api_type == "async" else "LATENCY"
                    logger.warning(f"PerformanceMode was not explicitly specified in command line. " +
                    f"Device {device} performance hint will be set to " + args.perf_hint + ".")
            else:
                logger.warning(f"Device {device} does not support performance hint property(-hint).")

        version = benchmark.get_version_info()

        logger.info(version)

        # --------------------- 3. Setting device configuration --------------------------------------------------------
        next_step()
        def get_device_type_from_name(name) :
            new_name = str(name)
            new_name = new_name.split(".", 1)[0]
            new_name = new_name.split("(", 1)[0]
            return new_name

        ## Set default values from dumped config
        default_devices = set()
        for device in devices:
            device_type = get_device_type_from_name(device)
            if device_type in config and device not in config:
                config[device] = config[device_type].copy()
                default_devices.add(device_type)

        for def_device in default_devices:
            config.pop(def_device)

        perf_counts = False
        for device in devices:
            supported_properties = benchmark.core.get_property(device, 'SUPPORTED_PROPERTIES')
            if device not in config.keys():
                config[device] = {}
            ## Set performance counter
            if is_flag_set_in_command_line('pc'):
                ## set to user defined value
                config[device]['PERF_COUNT'] = 'YES' if args.perf_counts else 'NO'
            elif 'PERF_COUNT' in config[device].keys() and config[device]['PERF_COUNT'] == 'YES':
                logger.warning(f"Performance counters for {device} device is turned on. " +
                               "To print results use -pc option.")
            elif args.report_type in [ averageCntReport, detailedCntReport ]:
                logger.warning(f"Turn on performance counters for {device} device " +
                               f"since report type is {args.report_type}.")
                config[device]['PERF_COUNT'] = 'YES'
            elif args.exec_graph_path is not None:
                logger.warning(f"Turn on performance counters for {device} device " +
                               "due to execution graph dumping.")
                config[device]['PERF_COUNT'] = 'YES'
            else:
                ## set to default value
                config[device]['PERF_COUNT'] = 'YES' if args.perf_counts else 'NO'
            perf_counts = True if config[device]['PERF_COUNT'] == 'YES' else perf_counts

            ## high-level performance hints
            if is_flag_set_in_command_line('hint') or args.perf_hint:
                config[device]['PERFORMANCE_HINT'] = args.perf_hint.upper()
                if is_flag_set_in_command_line('nireq'):
                    config[device]['PERFORMANCE_HINT_NUM_REQUESTS'] = str(args.number_infer_requests)

            ## infer precision
            if device in device_infer_precision and 'INFERENCE_PRECISION_HINT' in supported_properties:
                config[device]['INFERENCE_PRECISION_HINT'] = device_infer_precision[device]
            elif device in device_infer_precision:
                raise Exception(f"Device {device} doesn't support config key INFERENCE_PRECISION_HINT!" \
                                " Please specify -infer_precision for correct devices in format" \
                                " <dev1>:<infer_precision1>,<dev2>:<infer_precision2> or via configuration file.")

            ## the rest are individual per-device settings (overriding the values the device will deduce from perf hint)
            def set_throughput_streams():
                key = get_device_type_from_name(device) + "_THROUGHPUT_STREAMS"
                if device in device_number_streams.keys():
                    ## set to user defined value
                    if key in supported_properties:
                        config[device][key] = device_number_streams[device]
                    elif "NUM_STREAMS" in supported_properties:
                        key = "NUM_STREAMS"
                        config[device][key] = device_number_streams[device]
                    else:
                        raise Exception(f"Device {device} doesn't support config key '{key}'! " +
                                        "Please specify -nstreams for correct devices in format  <dev1>:<nstreams1>,<dev2>:<nstreams2>")
                elif key not in config[device].keys() and args.api_type == "async" \
                    and 'PERFORMANCE_HINT' in config[device].keys() and config[device]['PERFORMANCE_HINT'] == '':
                    ## set the _AUTO value for the #streams
                    logger.warning(f"-nstreams default value is determined automatically for {device} device. " +
                                   "Although the automatic selection usually provides a reasonable performance, "
                                   "but it still may be non-optimal for some cases, for more information look at README.")
                    if device != MYRIAD_DEVICE_NAME:  ## MYRIAD sets the default number of streams implicitly
                        if key in supported_properties:
                            config[device][key] = get_device_type_from_name(device) + "_THROUGHPUT_AUTO"
                        elif "NUM_STREAMS" in supported_properties:
                            key = "NUM_STREAMS"
                            config[device][key] = "-1"  # Set AUTO mode for streams number
                if key in config[device].keys():
                    device_number_streams[device] = config[device][key]

            if CPU_DEVICE_NAME in device: # CPU supports few special performance-oriented keys
                # limit threading for CPU portion of inference
                if args.number_threads and is_flag_set_in_command_line("nthreads"):
                    config[device]['CPU_THREADS_NUM'] = str(args.number_threads)

                if is_flag_set_in_command_line('pin'):
                    ## set to user defined value
                    config[device]['CPU_BIND_THREAD'] = args.infer_threads_pinning
                elif 'CPU_BIND_THREAD' not in config[device].keys():
                    if MULTI_DEVICE_NAME in device_name and GPU_DEVICE_NAME in device_name:
                        logger.warning(f"Turn off threads pinning for {device} " +
                                       "device since multi-scenario with GPU device is used.")
                        config[device]['CPU_BIND_THREAD'] = 'NO'

                ## for CPU execution, more throughput-oriented execution via streams
                set_throughput_streams()
            elif GPU_DEVICE_NAME in device:
                ## for GPU execution, more throughput-oriented execution via streams
                set_throughput_streams()

                if MULTI_DEVICE_NAME in device_name and CPU_DEVICE_NAME in device_name:
                    logger.warning("Turn on GPU throttling. Multi-device execution with the CPU + GPU performs best with GPU throttling hint, " +
                                   "which releases another CPU thread (that is otherwise used by the GPU driver for active polling)")
                    config[device]['GPU_PLUGIN_THROTTLE'] = '1'
            elif MYRIAD_DEVICE_NAME in device:
                set_throughput_streams()
                config[device]['LOG_LEVEL'] = 'LOG_INFO'
            else:
                if 'CPU_THREADS_NUM' in supported_properties and args.number_threads and is_flag_set_in_command_line("nthreads"):
                    config[device]['CPU_THREADS_NUM'] = str(args.number_threads)
                if 'CPU_THROUGHPUT_STREAMS' in supported_properties and args.number_streams and is_flag_set_in_command_line("streams"):
                    config[device]['CPU_THROUGHPUT_STREAMS'] = args.number_streams
                if 'CPU_BIND_THREAD' in supported_properties and args.infer_threads_pinning and is_flag_set_in_command_line("pin"):
                    config[device]['CPU_BIND_THREAD'] = args.infer_threads_pinning
        perf_counts = perf_counts
        benchmark.set_config(config)
        if args.cache_dir:
            benchmark.set_cache_dir(args.cache_dir)

        ## If set batch size, disable the auto batching
        if args.batch_size:
            benchmark.set_allow_auto_batching(False)

        topology_name = ""
        load_from_file_enabled = is_flag_set_in_command_line('load_from_file') or is_flag_set_in_command_line('lfile')
        if load_from_file_enabled and not is_network_compiled:
            next_step()
            print("Skipping the step for loading model from file")
            next_step()
            print("Skipping the step for loading model from file")
            next_step()
            print("Skipping the step for loading model from file")

            # --------------------- 7. Loading the model to the device -------------------------------------------------
            next_step()

            start_time = datetime.utcnow()
            compiled_model = benchmark.core.compile_model(args.path_to_model, benchmark.device)
            duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
            logger.info(f"Compile model took {duration_ms} ms")
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('load network time (ms)', duration_ms)
                                          ])
            app_inputs_info, _ = get_inputs_info(args.shape, args.data_shape, args.layout, args.batch_size, args.input_scale, args.input_mean, compiled_model.inputs)
            batch_size = get_network_batch_size(app_inputs_info)
        elif not is_network_compiled:
            # --------------------- 4. Read the Intermediate Representation of the network -----------------------------
            next_step()

            start_time = datetime.utcnow()
            model = benchmark.read_model(args.path_to_model)
            topology_name = model.get_name()
            duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
            logger.info(f"Read model took {duration_ms} ms")
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('read network time (ms)', duration_ms)
                                          ])

            # --------------------- 5. Resizing network to match image sizes and given batch ---------------------------
            next_step()

            app_inputs_info, reshape = get_inputs_info(args.shape, args.data_shape, args.layout, args.batch_size, args.input_scale, args.input_mean, model.inputs)
            if reshape:
                start_time = datetime.utcnow()
                shapes = { info.name : info.partial_shape for info in app_inputs_info }
                logger.info(
                    'Reshaping model: {}'.format(', '.join("'{}': {}".format(k, str(v)) for k, v in shapes.items())))
                model.reshape(shapes)
                duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
                logger.info(f"Reshape model took {duration_ms} ms")
                if statistics:
                    statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                              [
                                                  ('reshape network time (ms)', duration_ms)
                                              ])

            # use batch size according to provided layout and shapes
            batch_size = get_network_batch_size(app_inputs_info)
            logger.info(f'Network batch size: {batch_size}')

            # --------------------- 6. Configuring inputs and outputs of the model --------------------------------------------------
            next_step()

            pre_post_processing(model, app_inputs_info, args.input_precision, args.output_precision, args.input_output_precision)
            print_inputs_and_outputs_info(model)

            # --------------------- 7. Loading the model to the device -------------------------------------------------
            next_step()

            start_time = datetime.utcnow()
            compiled_model = benchmark.core.compile_model(model, benchmark.device)
            duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
            logger.info(f"Compile model took {duration_ms} ms")
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('load network time (ms)', duration_ms)
                                          ])
        else:
            next_step()
            print("Skipping the step for compiled network")
            next_step()
            print("Skipping the step for compiled network")
            next_step()
            print("Skipping the step for compiled network")

            # --------------------- 7. Loading the model to the device -------------------------------------------------
            next_step()

            start_time = datetime.utcnow()
            compiled_model = benchmark.core.import_model(args.path_to_model)
            duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
            logger.info(f"Import model took {duration_ms} ms")
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('import network time (ms)', duration_ms)
                                          ])
            app_inputs_info, _ = get_inputs_info(args.shape, args.data_shape, args.layout, args.batch_size, args.input_scale, args.input_mean, compiled_model.inputs)
            batch_size = get_network_batch_size(app_inputs_info)

        # --------------------- 8. Querying optimal runtime parameters --------------------------------------------------
        next_step()
        ## actual device-deduced settings
        for device in devices:
            keys = benchmark.core.get_property(device, 'SUPPORTED_PROPERTIES')
            logger.info(f'DEVICE: {device}')
            for k in keys:
                if k not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
                    try:
                        logger.info(f'  {k}  , {benchmark.core.get_property(device, k)}')
                    except:
                        pass


        # Update number of streams
        for device in device_number_streams.keys():
            try:
                key = get_device_type_from_name(device) + '_THROUGHPUT_STREAMS'
                device_number_streams[device] = benchmark.core.get_property(device, key)
            except:
                key = 'NUM_STREAMS'
                device_number_streams[device] = benchmark.core.get_property(device, key)

        # ------------------------------------ 9. Creating infer requests and preparing input data ----------------------
        next_step()

        # Create infer requests
        start_time = datetime.utcnow()
        requests = benchmark.create_infer_requests(compiled_model)
        duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
        logger.info(f"Create {benchmark.nireq} infer requests took {duration_ms} ms")
        if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('create infer requests time (ms)', duration_ms)
                                          ])

        # Prepare input data
        paths_to_input = list()
        if args.paths_to_input:
            for path in args.paths_to_input:
                if ":" in next(iter(path), ""):
                    paths_to_input.extend(path)
                else:
                    paths_to_input.append(os.path.abspath(*path))

        data_queue = get_input_data(paths_to_input, app_inputs_info)

        static_mode = check_for_static(app_inputs_info)
        allow_inference_only_or_sync = can_measure_as_static(app_inputs_info)
        if not allow_inference_only_or_sync and benchmark.api_type == 'sync':
            raise Exception("Benchmarking of the model with dynamic shapes is available for async API only."
                                   "Please use -api async -nstreams 1 -nireq 1 to emulate sync behavior.")

        if benchmark.inference_only == None:
            if static_mode:
                benchmark.inference_only = True
            else:
                benchmark.inference_only = False
        elif benchmark.inference_only and not allow_inference_only_or_sync:
            raise Exception("Benchmarking dynamic model available with input filling in measurement loop only!")

        # update batch size in case dynamic network with one data_shape
        if benchmark.inference_only and batch_size.is_dynamic:
            batch_size = Dimension(data_queue.batch_sizes[data_queue.current_group_id])

        benchmark.latency_groups = get_latency_groups(app_inputs_info)

        if len(benchmark.latency_groups) > 1:
            logger.info(f"Defined {len(benchmark.latency_groups)} tensor groups:")
            for group in benchmark.latency_groups:
                print(f"\t{str(group)}")

        # Iteration limit
        benchmark.niter = get_number_iterations(benchmark.niter, benchmark.nireq, max(len(info.shapes) for info in app_inputs_info), benchmark.api_type)

        # Set input tensors before first inference
        for request in requests:
            data_tensors = data_queue.get_next_input()
            for port, data_tensor in data_tensors.items():
                input_tensor = request.get_input_tensor(port)
                if not static_mode:
                    input_tensor.shape = data_tensor.shape
                if not len(input_tensor.shape):
                    input_tensor.data.flat[:] = data_tensor.data
                else:
                    input_tensor.data[:] = data_tensor.data

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                      [
                                          ('topology', topology_name),
                                          ('target device', device_name),
                                          ('API', args.api_type),
                                          ('inference_only', benchmark.inference_only),
                                          ('precision', "UNSPECIFIED"),
                                          ('batch size', str(batch_size)),
                                          ('number of iterations', str(benchmark.niter)),
                                          ('number of parallel infer requests', str(benchmark.nireq)),
                                          ('duration (ms)', str(get_duration_in_milliseconds(benchmark.duration_seconds))),
                                       ])

            for nstreams in device_number_streams.items():
                statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                         [
                                            (f"number of {nstreams[0]} streams", str(nstreams[1])),
                                         ])

        # ------------------------------------ 10. Measuring performance -----------------------------------------------

        output_string = process_help_inference_string(benchmark, device_number_streams)

        next_step(additional_info=output_string)

        if benchmark.inference_only:
            logger.info("Benchmarking in inference only mode (inputs filling are not included in measurement loop).")
        else:
            logger.info("Benchmarking in full mode (inputs filling are included in measurement loop).")

        progress_bar_total_count = 10000
        if benchmark.niter and not benchmark.duration_seconds:
            progress_bar_total_count = benchmark.niter

        progress_bar = ProgressBar(progress_bar_total_count, args.stream_output, args.progress) if args.progress else None

        duration_ms = f"{benchmark.first_infer(requests):.2f}"
        logger.info(f"First inference took {duration_ms} ms")
        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                    [
                                        ('first inference time (ms)', duration_ms)
                                    ])

        pcseq = args.pcseq
        if static_mode or len(benchmark.latency_groups) == 1:
            pcseq = False

        fps, median_latency_ms, avg_latency_ms, min_latency_ms, max_latency_ms, total_duration_sec, iteration = benchmark.main_loop(requests, data_queue, batch_size, args.latency_percentile, progress_bar, pcseq)

        # ------------------------------------ 11. Dumping statistics report -------------------------------------------
        next_step()

        if args.dump_config:
            dump_config(args.dump_config, config)
            logger.info(f"OpenVINO configuration settings were dumped to {args.dump_config}")

        if args.exec_graph_path:
            dump_exec_graph(compiled_model, args.exec_graph_path)

        if perf_counts:
            perfs_count_list = []
            for request in requests:
                perfs_count_list.append(request.profiling_info)
            if args.perf_counts:
                print_perf_counters(perfs_count_list)
            if statistics:
                statistics.dump_performance_counters(perfs_count_list)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('total execution time (ms)', f'{get_duration_in_milliseconds(total_duration_sec):.2f}'),
                                          ('total number of iterations', str(iteration)),
                                      ])
            if MULTI_DEVICE_NAME not in device_name:
                latency_prefix = None
                if args.latency_percentile == 50 and static_mode:
                    #latency_prefix = 'median latency (ms)'
                    latency_prefix = 'latency (ms)'
                elif args.latency_percentile != 50:
                    latency_prefix = 'latency (' + str(args.latency_percentile) + ' percentile) (ms)'
                if latency_prefix:
                    statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                            [
                                                (latency_prefix, f'{median_latency_ms:.2f}'),
                                            ])
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("avg latency", f'{avg_latency_ms:.2f}'),
                                          ])
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("min latency", f'{min_latency_ms:.2f}'),
                                          ])
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("max latency", f'{max_latency_ms:.2f}'),
                                          ])
                if pcseq:
                    for group in benchmark.latency_groups:
                        statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("group", str(group)),
                                          ])
                        statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("avg latency", f'{group.avg:.2f}'),
                                          ])
                        statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("min latency", f'{group.min:.2f}'),
                                          ])
                        statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("max latency", f'{group.max:.2f}'),
                                          ])
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('throughput', f'{fps:.2f}'),
                                      ])
            statistics.dump()


        print(f'Count:          {iteration} iterations')
        print(f'Duration:       {get_duration_in_milliseconds(total_duration_sec):.2f} ms')
        if MULTI_DEVICE_NAME not in device_name:
            print('Latency:')
            if args.latency_percentile == 50 and static_mode:
                print(f'    Median:     {median_latency_ms:.2f} ms')
            elif args.latency_percentile != 50:
                print(f'({args.latency_percentile} percentile):     {median_latency_ms:.2f} ms')
            print(f'    AVG:        {avg_latency_ms:.2f} ms')
            print(f'    MIN:        {min_latency_ms:.2f} ms')
            print(f'    MAX:        {max_latency_ms:.2f} ms')

            if pcseq:
                print("Latency for each data shape group: ")
                for group in benchmark.latency_groups:
                    print(f"  {str(group)}")
                    print(f'    AVG:        {group.avg:.2f} ms')
                    print(f'    MIN:        {group.min:.2f} ms')
                    print(f'    MAX:        {group.max:.2f} ms')

        print(f'Throughput: {fps:.2f} FPS')

        del compiled_model

        next_step.step_id = 0
    except Exception as e:
        logger.exception(e)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('error', str(e)),
                                      ])
            statistics.dump()
        sys.exit(1)
