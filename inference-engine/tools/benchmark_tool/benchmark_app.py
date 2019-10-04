import os
import sys
from datetime import datetime

from parameters import parse_args
from openvino.tools.benchmark.benchmark import Benchmark
from openvino.tools.benchmark.utils.constants import MULTI_DEVICE_NAME
from openvino.tools.benchmark.utils.infer_request_wrap import InferRequestsQueue
from openvino.tools.benchmark.utils.inputs_filling import get_inputs
from openvino.tools.benchmark.utils.logging import logger
from openvino.tools.benchmark.utils.progress_bar import ProgressBar
from openvino.tools.benchmark.utils.utils import next_step, read_network, config_network_inputs, get_number_iterations, \
    process_help_inference_string, print_perf_counters, dump_exec_graph, get_duration_in_milliseconds, \
    get_command_line_arguments
from openvino.tools.benchmark.utils.statistics_report import StatisticsReport, averageCntReport, detailedCntReport

def main(args):
    statistics = None
    try:
        if args.number_streams is None:
                logger.warn(" -nstreams default value is determined automatically for a device. "
                            "Although the automatic selection usually provides a reasonable performance, "
                            "but it still may be non-optimal for some cases, for more information look at README. ")

        if args.report_type:
          statistics = StatisticsReport(StatisticsReport.Config(args.report_type, args.report_folder))
          statistics.add_parameters(StatisticsReport.Category.COMMAND_LINE_PARAMETERS, get_command_line_arguments(sys.argv))


        # ------------------------------ 2. Loading Inference Engine ---------------------------------------------------
        next_step()

        device_name = args.target_device.upper()

        benchmark = Benchmark(args.target_device, args.number_infer_requests,
                              args.number_iterations, args.time, args.api_type)

        benchmark.add_extension(args.path_to_extension, args.path_to_cldnn_config)

        version = benchmark.get_version_info()

        logger.info(version)

        # --------------------- 3. Read the Intermediate Representation of the network ---------------------------------
        next_step()

        start_time = datetime.now()
        ie_network = read_network(args.path_to_model)
        duration_ms = "{:.2f}".format((datetime.now() - start_time).total_seconds() * 1000)
        if statistics:
            logger.info("Read network took {} ms".format(duration_ms))
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('read network time (ms)', duration_ms)
                                      ])

        # --------------------- 4. Resizing network to match image sizes and given batch -------------------------------

        next_step()
        if args.batch_size and args.batch_size != ie_network.batch_size:
            benchmark.reshape(ie_network, args.batch_size)
        batch_size = ie_network.batch_size
        logger.info('Network batch size: {}, precision: {}'.format(ie_network.batch_size, ie_network.precision))

        # --------------------- 5. Configuring input of the model ------------------------------------------------------
        next_step()

        config_network_inputs(ie_network)

        # --------------------- 6. Setting device configuration --------------------------------------------------------
        next_step()
        benchmark.set_config(args.number_streams, args.api_type, args.number_threads,
                             args.infer_threads_pinning)

        # --------------------- 7. Loading the model to the device -----------------------------------------------------
        next_step()

        start_time = datetime.now()
        perf_counts = True if args.perf_counts or \
                              args.report_type in [ averageCntReport, detailedCntReport ] or \
                              args.exec_graph_path else False
        exe_network = benchmark.load_network(ie_network, perf_counts, args.number_infer_requests)
        duration_ms = "{:.2f}".format((datetime.now() - start_time).total_seconds() * 1000)
        if statistics:
            logger.info("Load network took {} ms".format(duration_ms))
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('load network time (ms)', duration_ms)
                                      ])

        # --------------------- 8. Setting optimal runtime parameters --------------------------------------------------
        next_step()

        # Number of requests
        infer_requests = exe_network.requests
        benchmark.nireq = len(infer_requests)

        # Iteration limit
        benchmark.niter = get_number_iterations(benchmark.niter, len(exe_network.requests), args.api_type)

        # ------------------------------------ 9. Creating infer requests and filling input blobs ----------------------
        next_step()

        request_queue = InferRequestsQueue(infer_requests)

        path_to_input = os.path.abspath(args.path_to_input) if args.path_to_input else None
        requests_input_data = get_inputs(path_to_input, batch_size, ie_network.inputs, infer_requests)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                      [
                                          ('topology', ie_network.name),
                                          ('target device', device_name),
                                          ('API', args.api_type),
                                          ('precision', str(ie_network.precision)),
                                          ('batch size', str(ie_network.batch_size)),
                                          ('number of iterations', str(benchmark.niter) if benchmark.niter else "0"),
                                          ('number of parallel infer requests', str(benchmark.nireq)),
                                          ('duration (ms)', str(get_duration_in_milliseconds(benchmark.duration_seconds))),
                                       ])

            for nstreams in benchmark.device_number_streams.items():
                statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                         [
                                            ("number of {} streams".format(nstreams[0]), str(nstreams[1])),
                                         ])

        # ------------------------------------ 10. Measuring performance -----------------------------------------------

        output_string = process_help_inference_string(benchmark)

        next_step(output_string)
        progress_bar_total_count = 10000
        if benchmark.niter and not benchmark.duration_seconds:
            progress_bar_total_count = benchmark.niter

        progress_bar = ProgressBar(progress_bar_total_count, args.stream_output, args.progress)

        fps, latency_ms, total_duration_sec, iteration = benchmark.infer(request_queue, requests_input_data,
                                                                         batch_size, progress_bar)

        # ------------------------------------ 11. Dumping statistics report -------------------------------------------
        next_step()

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

if __name__ == "__main__":
    # ------------------------------ 1. Parsing and validating input arguments -------------------------------------
    next_step()

    main(parse_args())
