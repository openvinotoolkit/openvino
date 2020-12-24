import os
import sys
from datetime import datetime

from openvino.tools.benchmark.benchmark import Benchmark
from openvino.tools.benchmark.parameters import parse_args
from openvino.tools.benchmark.utils.constants import MULTI_DEVICE_NAME, HETERO_DEVICE_NAME, CPU_DEVICE_NAME, \
    GPU_DEVICE_NAME, MYRIAD_DEVICE_NAME, GNA_DEVICE_NAME, BLOB_EXTENSION
from openvino.tools.benchmark.utils.inputs_filling import set_inputs
from openvino.tools.benchmark.utils.logging import logger
from openvino.tools.benchmark.utils.progress_bar import ProgressBar
from openvino.tools.benchmark.utils.utils import next_step, config_network_inputs, get_number_iterations, \
    process_help_inference_string, print_perf_counters, dump_exec_graph, get_duration_in_milliseconds, \
    get_command_line_arguments, parse_nstreams_value_per_device, parse_devices, update_shapes, \
    adjust_shapes_batch, load_config, dump_config
from openvino.tools.benchmark.utils.statistics_report import StatisticsReport, averageCntReport, detailedCntReport


def main():
    # ------------------------------ 1. Parsing and validating input arguments -------------------------------------
    next_step()
    run(parse_args())


def run(args):
    statistics = None
    try:
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
        device_number_streams = parse_nstreams_value_per_device(devices, args.number_streams)

        config = {}
        if args.load_config:
            load_config(args.load_config, config)

        is_network_compiled = False
        _, ext = os.path.splitext(args.path_to_model)

        if ext == BLOB_EXTENSION:
            is_network_compiled = True
            print("Network is compiled")

        # ------------------------------ 2. Loading Inference Engine ---------------------------------------------------
        next_step(step_id=2)

        benchmark = Benchmark(args.target_device, args.number_infer_requests,
                              args.number_iterations, args.time, args.api_type)

        ## CPU (MKLDNN) extensions
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

        version = benchmark.get_version_info()

        logger.info(version)

        # --------------------- 3. Setting device configuration --------------------------------------------------------
        next_step()

        perf_counts = False
        for device in devices:
            if device not in config.keys():
                config[device] = {}
            ## Set performance counter
            if is_flag_set_in_command_line('pc'):
                ## set to user defined value
                config[device]['PERF_COUNT'] = 'YES' if args.perf_counts else 'NO'
            elif 'PERF_COUNT' in config[device].keys() and config[device]['PERF_COUNT'] == 'YES':
                logger.warning("Performance counters for {} device is turned on. ".format(device) +
                               "To print results use -pc option.")
            elif args.report_type in [ averageCntReport, detailedCntReport ]:
                logger.warning("Turn on performance counters for {} device ".format(device) +
                               "since report type is {}.".format(args.report_type))
                config[device]['PERF_COUNT'] = 'YES'
            elif args.exec_graph_path is not None:
                logger.warning("Turn on performance counters for {} device ".format(device) +
                               "due to execution graph dumping.")
                config[device]['PERF_COUNT'] = 'YES'
            else:
                ## set to default value
                config[device]['PERF_COUNT'] = 'YES' if args.perf_counts else 'NO'
            perf_counts = True if config[device]['PERF_COUNT'] == 'YES' else perf_counts

            def set_throughput_streams():
                key = device + "_THROUGHPUT_STREAMS"
                if device in device_number_streams.keys():
                    ## set to user defined value
                    supported_config_keys = benchmark.ie.get_metric(device, 'SUPPORTED_CONFIG_KEYS')
                    if key not in supported_config_keys:
                        raise Exception("Device {} doesn't support config key '{}'! ".format(device, key) +
                                        "Please specify -nstreams for correct devices in format  <dev1>:<nstreams1>,<dev2>:<nstreams2>")
                    config[device][key] = device_number_streams[device]
                elif key not in config[device].keys() and args.api_type == "async":
                    logger.warning("-nstreams default value is determined automatically for {} device. ".format(device) +
                                   "Although the automatic selection usually provides a reasonable performance,"
                                   "but it still may be non-optimal for some cases, for more information look at README.")
                    config[device][key] = device + "_THROUGHPUT_AUTO"
                if key in config[device].keys():
                    device_number_streams[device] = config[device][key]

            if device == CPU_DEVICE_NAME: # CPU supports few special performance-oriented keys
                # limit threading for CPU portion of inference
                if args.number_threads and is_flag_set_in_command_line("nthreads"):
                    config[device]['CPU_THREADS_NUM'] = str(args.number_threads)

                if is_flag_set_in_command_line("enforcebf16") or is_flag_set_in_command_line("enforce_bfloat16"):
                    config[device]['ENFORCE_BF16'] = 'YES' if args.enforce_bfloat16 else 'NO'

                if is_flag_set_in_command_line('pin'):
                    ## set to user defined value
                    config[device]['CPU_BIND_THREAD'] = args.infer_threads_pinning
                elif 'CPU_BIND_THREAD' not in config[device].keys():
                    if MULTI_DEVICE_NAME in device_name and GPU_DEVICE_NAME in device_name:
                        logger.warning("Turn off threads pinning for {}".format(device) +
                                       "device since multi-scenario with GPU device is used.")
                        config[device]['CPU_BIND_THREAD'] = 'NO'
                    else:
                        ## set to default value
                        config[device]['CPU_BIND_THREAD'] = args.infer_threads_pinning

                ## for CPU execution, more throughput-oriented execution via streams
                set_throughput_streams()
            elif device == GPU_DEVICE_NAME:
                ## for GPU execution, more throughput-oriented execution via streams
                set_throughput_streams()

                if MULTI_DEVICE_NAME in device_name and CPU_DEVICE_NAME in device_name:
                    logger.warning("Turn on GPU trottling. Multi-device execution with the CPU + GPU performs best with GPU trottling hint, " +
                                   "which releases another CPU thread (that is otherwise used by the GPU driver for active polling)")
                    config[device]['CLDNN_PLUGIN_THROTTLE'] = '1'
            elif device == MYRIAD_DEVICE_NAME:
                config[device]['LOG_LEVEL'] = 'LOG_INFO'
            elif device == GNA_DEVICE_NAME:
                if is_flag_set_in_command_line('qb'):
                    if args.qb == 8:
                        config[device]['GNA_PRECISION'] = 'I8'
                    else:
                        config[device]['GNA_PRECISION'] = 'I16'
                if args.number_threads and is_flag_set_in_command_line("nthreads"):
                    config[device]['GNA_LIB_N_THREADS'] = str(args.number_threads)
            else:
                supported_config_keys = benchmark.ie.get_metric(device, 'SUPPORTED_CONFIG_KEYS')
                if 'CPU_THREADS_NUM' in supported_config_keys and args.number_threads and is_flag_set_in_command_line("nthreads"):
                    config[device]['CPU_THREADS_NUM'] = str(args.number_threads)
                if 'CPU_THROUGHPUT_STREAMS' in supported_config_keys and args.number_streams and is_flag_set_in_command_line("streams"):
                    config[device]['CPU_THROUGHPUT_STREAMS'] = args.number_streams
                if 'CPU_BIND_THREAD' in supported_config_keys and args.infer_threads_pinning and is_flag_set_in_command_line("pin"):
                    config[device]['CPU_BIND_THREAD'] = args.infer_threads_pinning
        perf_counts = perf_counts

        benchmark.set_config(config)
        batch_size = args.batch_size
        if not is_network_compiled:
            # --------------------- 4. Read the Intermediate Representation of the network -----------------------------
            next_step()

            start_time = datetime.utcnow()
            ie_network = benchmark.read_network(args.path_to_model)
            duration_ms = "{:.2f}".format((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.info("Read network took {} ms".format(duration_ms))
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('read network time (ms)', duration_ms)
                                          ])

            # --------------------- 5. Resizing network to match image sizes and given batch ---------------------------
            next_step()

            shapes = {k: v.input_data.shape.copy() for k, v in ie_network.input_info.items()}
            reshape = False
            if args.shape:
                reshape |= update_shapes(shapes, args.shape, ie_network.input_info)
            if args.batch_size and args.batch_size != ie_network.batch_size:
                reshape |= adjust_shapes_batch(shapes, args.batch_size, ie_network.input_info)

            if reshape:
                start_time = datetime.utcnow()
                logger.info(
                    'Reshaping network: {}'.format(', '.join("'{}': {}".format(k, v) for k, v in shapes.items())))
                ie_network.reshape(shapes)
                duration_ms = "{:.2f}".format((datetime.utcnow() - start_time).total_seconds() * 1000)
                logger.info("Reshape network took {} ms".format(duration_ms))
                if statistics:
                    statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                              [
                                                  ('reshape network time (ms)', duration_ms)
                                              ])

            batch_size = ie_network.batch_size
            logger.info('Network batch size: {}'.format(ie_network.batch_size))

            # --------------------- 6. Configuring input of the model --------------------------------------------------
            next_step()

            config_network_inputs(ie_network)

            # --------------------- 7. Loading the model to the device -------------------------------------------------
            next_step()

            start_time = datetime.utcnow()
            exe_network = benchmark.load_network(ie_network)
            duration_ms = "{:.2f}".format((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.info("Load network took {} ms".format(duration_ms))
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
            exe_network = benchmark.import_network(args.path_to_model)
            duration_ms = "{:.2f}".format((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.info("Import network took {} ms".format(duration_ms))
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('import network time (ms)', duration_ms)
                                          ])
            if batch_size == 0:
                batch_size = 1

        # --------------------- 8. Setting optimal runtime parameters --------------------------------------------------
        next_step()

        # Update number of streams
        for device in device_number_streams.keys():
            key = device + '_THROUGHPUT_STREAMS'
            device_number_streams[device] = benchmark.ie.get_config(device, key)

        # Number of requests
        infer_requests = exe_network.requests

        # Iteration limit
        benchmark.niter = get_number_iterations(benchmark.niter, benchmark.nireq, args.api_type)

        # ------------------------------------ 9. Creating infer requests and filling input blobs ----------------------
        next_step()

        paths_to_input = list()
        if args.paths_to_input:
            for path in args.paths_to_input:
                paths_to_input.append(os.path.abspath(*path) if args.paths_to_input else None)
        set_inputs(paths_to_input, batch_size, exe_network.input_info, infer_requests)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                      [
                                          ('topology', ie_network.name),
                                          ('target device', device_name),
                                          ('API', args.api_type),
                                          ('precision', "UNSPECIFIED"),
                                          ('batch size', str(batch_size)),
                                          ('number of iterations', str(benchmark.niter) if benchmark.niter else "0"),
                                          ('number of parallel infer requests', str(benchmark.nireq)),
                                          ('duration (ms)', str(get_duration_in_milliseconds(benchmark.duration_seconds))),
                                       ])

            for nstreams in device_number_streams.items():
                statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                         [
                                            ("number of {} streams".format(nstreams[0]), str(nstreams[1])),
                                         ])

        # ------------------------------------ 10. Measuring performance -----------------------------------------------

        output_string = process_help_inference_string(benchmark)

        next_step(additional_info=output_string)
        progress_bar_total_count = 10000
        if benchmark.niter and not benchmark.duration_seconds:
            progress_bar_total_count = benchmark.niter

        progress_bar = ProgressBar(progress_bar_total_count, args.stream_output, args.progress) if args.progress else None

        duration_ms =  "{:.2f}".format(benchmark.first_infer(exe_network))
        logger.info("First inference took {} ms".format(duration_ms))
        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                    [
                                        ('first inference time (ms)', duration_ms)
                                    ])
        fps, latency_ms, total_duration_sec, iteration = benchmark.infer(exe_network, batch_size, progress_bar)

        # ------------------------------------ 11. Dumping statistics report -------------------------------------------
        next_step()

        if args.dump_config:
            dump_config(args.dump_config, config)
            logger.info("Inference Engine configuration settings were dumped to {}".format(args.dump_config))

        if args.exec_graph_path:
            dump_exec_graph(exe_network, args.exec_graph_path)

        if perf_counts:
            perfs_count_list = []
            for ni in range(int(benchmark.nireq)):
                perfs_count_list.append(exe_network.requests[ni].get_perf_counts())
            if args.perf_counts:
                print_perf_counters(perfs_count_list)
            if statistics:
              statistics.dump_performance_counters(perfs_count_list)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('total execution time (ms)', '{:.2f}'.format(get_duration_in_milliseconds(total_duration_sec))),
                                          ('total number of iterations', str(iteration)),
                                      ])
            if MULTI_DEVICE_NAME not in device_name:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('latency (ms)', '{:.2f}'.format(latency_ms)),
                                          ])

            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('throughput', '{:.2f}'.format(fps)),
                                      ])

        if statistics:
          statistics.dump()

        print('Count:      {} iterations'.format(iteration))
        print('Duration:   {:.2f} ms'.format(get_duration_in_milliseconds(total_duration_sec)))
        if MULTI_DEVICE_NAME not in device_name:
            print('Latency:    {:.2f} ms'.format(latency_ms))
        print('Throughput: {:.2f} FPS'.format(fps))

        del exe_network

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
