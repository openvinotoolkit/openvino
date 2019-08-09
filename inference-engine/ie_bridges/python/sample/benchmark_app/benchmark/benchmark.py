"""
 Copyright (C) 2018-2019 Intel Corporation

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

from statistics import median
from openvino.inference_engine import IENetwork, IECore, get_version

from .utils.parameters import *
from .utils.inputs_filling import *
from .utils.utils import *
from .utils.infer_request_wrap import *
from .utils.progress_bar import *

def getDurationInMilliseconds(duration):
    return duration * 1000

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(step_id = 0)
def next_step(additional_info = ""):
    step_names = {
        1  : "Parsing and validating input arguments",
        2  : "Loading Inference Engine",
        3  : "Read the Intermediate Representation of the network",
        4  : "Resizing network to match image sizes and given batch",
        5  : "Configuring input of the model",
        6  : "Setting device configuration",
        7  : "Loading the model to the device",
        8  : "Setting optimal runtime parameters",
        9  : "Creating infer requests and filling input blobs with images",
        10 : "Measuring performance",
        11 : "Dumping statistics report",
    }

    next_step.step_id += 1
    if (next_step.step_id not in step_names.keys()):
        raise Exception("Step ID " + str(next_step.step_id) + " is out of total steps number " + len(step_names))

    print("[Step {}/{}] {}".format(next_step.step_id, len(step_names), step_names[next_step.step_id]) + (" (" + additional_info + ")" if len(additional_info) else ""))

def main(args=None):
    try:
        # ------------------------------ 1. Parsing and validating input arguments -------------------------------------
        next_step()

        if not args:
            args = parse_args()

        # ------------------------------ 2. Loading Inference Engine ---------------------------------------------------
        next_step()

        device_name = args.target_device.upper()

        ie = IECore()

        if CPU_DEVICE_NAME in device_name:
            if args.path_to_extension:
                ie.add_extension(extension_path=args.path_to_extension, device_name=CPU_DEVICE_NAME)
        if GPU_DEVICE_NAME in device_name:
            if args.path_to_cldnn_config:
                ie.set_config({'CONFIG_FILE' : args.path_to_cldnn_config}, GPU_DEVICE_NAME)
                logger.info("GPU extensions is loaded {}".format(args.path_to_cldnn_config))

        logger.info("InferenceEngine:\n{: <9}{}".format("",get_version()))
        version_string = "Device is {}\n".format(device_name)
        for device, version in ie.get_versions(device_name).items():
          version_string += "{: <9}{}\n".format("", device)
          version_string += "{: <9}{:.<24}{} {}.{}\n".format("",version.description," version", version.major, version.minor)
          version_string += "{: <9}{:.<24} {}\n".format("","Build", version.build_number)
        logger.info(version_string)

        # --------------------- 3. Read the Intermediate Representation of the network ---------------------------------
        next_step()

        xml_filename = os.path.abspath(args.path_to_model)
        head, tail = os.path.splitext(xml_filename)
        bin_filename = os.path.abspath(head + BIN_EXTENSION)

        ie_network = IENetwork(xml_filename, bin_filename)

        input_info = ie_network.inputs

        if len(input_info) == 0:
            raise AttributeError('No inputs info is provided')

        # --------------------- 4. Resizing network to match image sizes and given batch -------------------------------
        next_step()

        batch_size = ie_network.batch_size
        precision = ie_network.precision

        if args.batch_size and args.batch_size != ie_network.batch_size:
            new_shapes = {}
            for key in input_info.keys():
                shape = input_info[key].shape
                layout = input_info[key].layout

                batchIndex = -1
                if ((layout == 'NCHW') or (layout == 'NCDHW') or
                    (layout == 'NHWC') or (layout == 'NDHWC') or
                    (layout == 'NC')):
                    batchIndex = 0
                elif (layout == 'CN'):
                    batchIndex = 1

                if ((batchIndex != -1) and (shape[batchIndex] != args.batch_size)):
                    shape[batchIndex] = args.batch_size
                    new_shapes[key] = shape

            if (len(new_shapes) > 0):
                logger.info("Resizing network to batch = {}".format(args.batch_size))
                ie_network.reshape(new_shapes)

            batch_size = args.batch_size

        logger.info("Network batch size: {}, precision {}".format(batch_size, precision))

        # --------------------- 5. Configuring input of the model ------------------------------------------------------
        next_step()

        for key in input_info.keys():
            if (isImage(input_info[key])):
                # Set the precision of input data provided by the user
                # Should be called before load of the network to the plugin
                input_info[key].precision = 'U8'

        # --------------------- 6. Setting device configuration --------------------------------------------------------
        next_step()

        devices = parseDevices(device_name)
        device_nstreams = parseValuePerDevice(devices, args.number_streams)
        for device in devices:
          if device == CPU_DEVICE_NAME: ## CPU supports few special performance-oriented keys
            ## limit threading for CPU portion of inference
            if args.number_threads:
              ie.set_config({'CPU_THREADS_NUM': str(args.number_threads)}, device)

            # pin threads for CPU portion of inference
            ie.set_config({'CPU_BIND_THREAD': args.infer_threads_pinning}, device)

            ## for CPU execution, more throughput-oriented execution via streams
            # for pure CPU execution, more throughput-oriented execution via streams
            if args.api_type == 'async':
                ie.set_config({'CPU_THROUGHPUT_STREAMS': str(device_nstreams.get(device))
                                                         if device in device_nstreams.keys()
                                                         else 'CPU_THROUGHPUT_AUTO' }, device)
            device_nstreams[device] = int(ie.get_config(device, 'CPU_THROUGHPUT_STREAMS'))

          elif device == GPU_DEVICE_NAME:
            if args.api_type == 'async':
                ie.set_config({'GPU_THROUGHPUT_STREAMS' : str(device_nstreams.get(device))
                                                          if device in device_nstreams.keys()
                                                          else 'GPU_THROUGHPUT_AUTO'}, device)
            device_nstreams[device] = int(ie.get_config(device, 'GPU_THROUGHPUT_STREAMS'))

          elif device == MYRIAD_DEVICE_NAME:
            ie.set_config({'LOG_LEVEL': 'LOG_INFO',
                           'VPU_LOG_LEVEL': 'LOG_WARNING'}, MYRIAD_DEVICE_NAME)

        # --------------------- 7. Loading the model to the device -----------------------------------------------------
        next_step()

        config = { 'PERF_COUNT' : ('YES' if args.perf_counts else 'NO')}

        exe_network = ie.load_network(ie_network,
                                      device_name,
                                      config=config,
                                      num_requests=args.number_infer_requests if args.number_infer_requests else 0)

        # --------------------- 8. Setting optimal runtime parameters --------------------------------------------------
        next_step()

        ## Number of requests
        infer_requests = exe_network.requests
        nireq = len(infer_requests)

        ## Iteration limit
        niter = args.number_iterations
        if niter and args.api_type == 'async':
          niter = (int)((niter + nireq - 1)/nireq)*nireq
          if (args.number_iterations != niter):
            logger.warn("Number of iterations was aligned by request number "
                        "from {} to {} using number of requests {}".format(args.number_iterations, niter, nireq))

        ## Time limit
        duration_seconds = 0
        if args.time:
          ## time limit
          duration_seconds = args.time
        elif not args.number_iterations:
          ## default time limit
          duration_seconds = get_duration_in_secs(device)

        # ------------------------------------ 8. Creating infer requests and filling input blobs ----------------------
        next_step()

        request_queue = InferRequestsQueue(infer_requests)

        path_to_input = os.path.abspath(args.path_to_input) if args.path_to_input else None
        requests_input_data = getInputs(path_to_input, batch_size, ie_network.inputs, infer_requests)

        # ------------------------------------ 9. Measuring performance ------------------------------------------------

        progress_count = 0
        progress_bar_total_count = 10000

        output_string = "Start inference {}ronously".format(args.api_type)
        if (args.api_type == "async"):
            if output_string != "":
                output_string += ", "

            output_string += str(nireq) + " inference requests"
            device_ss = ''
            for device, nstreams in device_nstreams.items():
                if device_ss != '':
                    device_ss += ', '
                device_ss += "{} streams for {}".format(str(nstreams), device)
            if device_ss != '':
                output_string += " using " + device_ss

        output_string += ", limits: "
        if niter:
            if not duration_seconds:
                progress_bar_total_count = niter
            output_string += str(niter) + " iterations"

        if duration_seconds:
            if niter:
                output_string += ", "
            output_string += str(getDurationInMilliseconds(duration_seconds)) + " ms duration"

        next_step(output_string)

        ## warming up - out of scope
        infer_request = request_queue.getIdleRequest()
        if not infer_request:
            raise Exception("No idle Infer Requests!")

        if (args.api_type == 'sync'):
            infer_request.infer(requests_input_data[infer_request.id])
        else:
            infer_request.startAsync(requests_input_data[infer_request.id])

        request_queue.waitAll()
        request_queue.resetTimes()

        start_time = datetime.now()
        exec_time = (datetime.now() - start_time).total_seconds()
        iteration = 0

        progress_bar = ProgressBar(progress_bar_total_count, args.stream_output, args.progress)

        ## Start inference & calculate performance
        ## to align number if iterations to guarantee that last infer requests are executed in the same conditions **/
        while ((niter and iteration < niter) or
               (duration_seconds and exec_time < duration_seconds) or
               (args.api_type == "async" and iteration % nireq != 0)):
            infer_request = request_queue.getIdleRequest()
            if not infer_request:
                raise Exception("No idle Infer Requests!")

            if (args.api_type == 'sync'):
                infer_request.infer(requests_input_data[infer_request.id])
            else:
                infer_request.startAsync(requests_input_data[infer_request.id])
            iteration += 1

            exec_time = (datetime.now() - start_time).total_seconds()

            if niter:
                progress_bar.add_progress(1)
            else:
                ## calculate how many progress intervals are covered by current iteration.
                ## depends on the current iteration time and time of each progress interval.
                ## Previously covered progress intervals must be skipped.
                progress_interval_time = duration_seconds / progress_bar_total_count
                new_progress = (int) (exec_time / progress_interval_time - progress_count)
                progress_bar.add_progress(new_progress)
                progress_count += new_progress

        ## wait the latest inference executions
        request_queue.waitAll()

        total_duration_sec = request_queue.getDurationInSeconds()
        times = request_queue.times
        times.sort()
        latency_ms = median(times)
        fps = batch_size * 1000 / latency_ms if args.api_type == 'sync' else batch_size * iteration / total_duration_sec

        progress_bar.finish()

        # ------------------------------------ 10. Dumping statistics report -------------------------------------------
        next_step()

        if args.exec_graph_path:
            try:
              exec_graph_info = exe_network.get_exec_graph_info()
              exec_graph_info.serialize(args.exec_graph_path)
              logger.info("Executable graph is stored to {}".format(args.exec_graph_path))
              del exec_graph_info
            except Exception as e:
                logging.exception(e)

        if args.perf_counts:
            for ni in range(int(nireq)):
                perf_counts = exe_network.requests[ni].get_perf_counts()
                logger.info("Pefrormance counts for {}-th infer request".format(ni))
                for layer, stats in perf_counts.items():
                    max_layer_name = 30
                    print("{:<30}{:<15}{:<30}{:<20}{:<20}{:<20}".format(layer[:max_layer_name - 4] + '...' if (len(layer) >= max_layer_name) else layer,
                                                                        stats['status'],
                                                                        'layerType: ' + str(stats['layer_type']),
                                                                        'realTime: ' + str(stats['real_time']),
                                                                        'cpu: ' + str(stats['cpu_time']),
                                                                        'execType: ' + str(stats['exec_type'])))

        print("Count:      {} iterations".format(iteration))
        print("Duration:   {:.2f} ms".format(getDurationInMilliseconds(total_duration_sec)))
        print("Latency:    {:.4f} ms".format(latency_ms))
        print("Throughput: {:.2f} FPS".format(fps))

        del exe_network
        del ie
        next_step.step_id = 0
    except Exception as e:
        logging.exception(e)
