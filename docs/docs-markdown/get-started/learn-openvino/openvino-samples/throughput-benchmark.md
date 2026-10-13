---
sidebar_position: 9
sidebar_label: "Throughput Benchmark Sample"
description: "Learn how to estimate performance of a model using Asynchronous Inference Request API in throughput mode (Python, C++)."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Throughput Benchmark Sample

This sample demonstrates how to estimate performance of a model using Asynchronous
Inference Request API in throughput mode. This sample
does not have other configurable command-line arguments. Feel free to modify sample's
source code to try out different options.

The reported results may deviate from what [benchmark_app](benchmark-tool.md)
reports. One example is model input precision for computer vision tasks. benchmark\_app
sets `uint8`, while the sample uses default model precision which is usually `float32`.

Before using the sample, refer to the following requirements:

- The sample accepts any file format supported by `core.read_model`.
- The sample has been validated with: yolo-v3-tf and face-detection-0200 models.
- To build the sample, use instructions available at [Build the Sample Applications](get-started-demos.md#build-samples)
  section in "Get Started with Samples" guide.

## How It Works

The sample compiles a model for a given device, randomly generates input data,
performs asynchronous inference multiple times for a given number of seconds.
Then, it processes and reports performance results.

<Tabs>
  <TabItem value="python" label="Python">
  <div style={{maxHeight: '400px', overflowY: 'auto'}}>

  ```python
  #!/usr/bin/env python3
  # -*- coding: utf-8 -*-
  # Copyright (C) 2018-2026 Intel Corporation
  # SPDX-License-Identifier: Apache-2.0

  import logging as log
  import sys
  import statistics
  from time import perf_counter

  import numpy as np
  import openvino as ov
  from openvino.utils.types import get_dtype


  def fill_tensor_random(tensor):
      dtype = get_dtype(tensor.element_type)
      rand_min, rand_max = (0, 1) if dtype == bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
      # np.random.uniform excludes high: add 1 to have it generated
      if np.dtype(dtype).kind in ['i', 'u', 'b']:
          rand_max += 1
      rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
      if 0 == tensor.get_size():
          raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference")
      tensor.data[:] = rs.uniform(rand_min, rand_max, list(tensor.shape)).astype(dtype)


  def main():
      log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
      log.info('OpenVINO:')
      log.info(f"{'Build ':.<39} {ov.__version__}")
      device_name = 'CPU'
      if len(sys.argv) == 3:
          device_name = sys.argv[2]
      elif len(sys.argv) != 2:
          log.info(f'Usage: {sys.argv[0]} <path_to_model> <device_name>(default: CPU)')
          return 1
      # Optimize for throughput. Best throughput can be reached by
      # running multiple openvino.InferRequest instances asyncronously
      tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}

      # Create Core and use it to compile a model.
      # Select the device by providing the name as the second parameter to CLI.
      # It is possible to set CUMULATIVE_THROUGHPUT as PERFORMANCE_HINT for AUTO device
      core = ov.Core()
      compiled_model = core.compile_model(sys.argv[1], device_name, tput)
      # AsyncInferQueue creates optimal number of InferRequest instances
      ireqs = ov.AsyncInferQueue(compiled_model)
      # Fill input data for ireqs
      for ireq in ireqs:
          for model_input in compiled_model.inputs:
              fill_tensor_random(ireq.get_tensor(model_input))
      # Warm up
      for _ in range(len(ireqs)):
          ireqs.start_async()
      ireqs.wait_all()
      # Benchmark for seconds_to_run seconds and at least niter iterations
      seconds_to_run = 10
      niter = 10
      latencies = []
      in_fly = set()
      start = perf_counter()
      time_point_to_finish = start + seconds_to_run
      while perf_counter() < time_point_to_finish or len(latencies) + len(in_fly) < niter:
          idle_id = ireqs.get_idle_request_id()
          if idle_id in in_fly:
              latencies.append(ireqs[idle_id].latency)
          else:
              in_fly.add(idle_id)
          ireqs.start_async()
      ireqs.wait_all()
      duration = perf_counter() - start
      for infer_request_id in in_fly:
          latencies.append(ireqs[infer_request_id].latency)
      # Report results
      fps = len(latencies) / duration
      log.info(f'Count:          {len(latencies)} iterations')
      log.info(f'Duration:       {duration * 1e3:.2f} ms')
      log.info('Latency:')
      log.info(f'    Median:     {statistics.median(latencies):.2f} ms')
      log.info(f'    Average:    {sum(latencies) / len(latencies):.2f} ms')
      log.info(f'    Min:        {min(latencies):.2f} ms')
      log.info(f'    Max:        {max(latencies):.2f} ms')
      log.info(f'Throughput: {fps:.2f} FPS')


  if __name__ == '__main__':
      main()
  ```

  </div>
  </TabItem>
  <TabItem value="cpp" label="C++">
  <div style={{maxHeight: '400px', overflowY: 'auto'}}>

  ```cpp
  // Copyright (C) 2018-2026 Intel Corporation
  // SPDX-License-Identifier: Apache-2.0
  //

  #include <algorithm>
  #include <condition_variable>
  #include <exception>
  #include <string>
  #include <vector>

  // clang-format off
  #include "openvino/openvino.hpp"

  #include "samples/args_helper.hpp"
  #include "samples/common.hpp"
  #include "samples/latency_metrics.hpp"
  #include "samples/slog.hpp"
  // clang-format on

  using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

  int main(int argc, char* argv[]) {
      try {
          slog::info << "OpenVINO:" << slog::endl;
          slog::info << ov::get_openvino_version();

          std::string device_name = "CPU";
          if (argc == 3) {
              device_name = argv[2];
          } else if (argc != 2) {
              slog::info << "Usage : " << argv[0] << " <path_to_model> <device_name>(default: CPU)" << slog::endl;
              return EXIT_FAILURE;
          }
          // Optimize for throughput. Best throughput can be reached by
          // running multiple ov::InferRequest instances asyncronously
          ov::AnyMap tput{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};

          // Create ov::Core and use it to compile a model.
          // Select the device by providing the name as the second parameter to CLI.
          // It is possible to set CUMULATIVE_THROUGHPUT as ov::hint::PerformanceMode for AUTO device
          ov::Core core;
          ov::CompiledModel compiled_model = core.compile_model(argv[1], device_name, tput);
          // Create optimal number of ov::InferRequest instances
          uint32_t nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
          std::vector<ov::InferRequest> ireqs(nireq);
          std::generate(ireqs.begin(), ireqs.end(), [&] {
              return compiled_model.create_infer_request();
          });
          // Fill input data for ireqs
          for (ov::InferRequest& ireq : ireqs) {
              for (const ov::Output<const ov::Node>& model_input : compiled_model.inputs()) {
                  fill_tensor_random(ireq.get_tensor(model_input));
              }
          }
          // Warm up
          for (ov::InferRequest& ireq : ireqs) {
              ireq.start_async();
          }
          for (ov::InferRequest& ireq : ireqs) {
              ireq.wait();
          }
          // Benchmark for seconds_to_run seconds and at least niter iterations
          std::chrono::seconds seconds_to_run{10};
          size_t niter = 10;
          std::vector<double> latencies;
          std::mutex mutex;
          std::condition_variable cv;
          std::exception_ptr callback_exception;
          struct TimedIreq {
              ov::InferRequest& ireq;  // ref
              std::chrono::steady_clock::time_point start;
              bool has_start_time;
          };
          std::deque<TimedIreq> finished_ireqs;
          for (ov::InferRequest& ireq : ireqs) {
              finished_ireqs.push_back({ireq, std::chrono::steady_clock::time_point{}, false});
          }
          auto start = std::chrono::steady_clock::now();
          auto time_point_to_finish = start + seconds_to_run;
          // Once there’s a finished ireq wake up main thread.
          // Compute and save latency for that ireq and prepare for next inference by setting up callback.
          // Callback pushes that ireq again to finished ireqs when infrence is completed.
          // Start asynchronous infer with updated callback
          for (;;) {
              std::unique_lock<std::mutex> lock(mutex);
              while (!callback_exception && finished_ireqs.empty()) {
                  cv.wait(lock);
              }
              if (callback_exception) {
                  std::rethrow_exception(callback_exception);
              }
              if (!finished_ireqs.empty()) {
                  auto time_point = std::chrono::steady_clock::now();
                  if (time_point > time_point_to_finish && latencies.size() > niter) {
                      break;
                  }
                  TimedIreq timedIreq = finished_ireqs.front();
                  finished_ireqs.pop_front();
                  if (lock.owns_lock()) {
                      lock.unlock();
                  }
                  ov::InferRequest& ireq = timedIreq.ireq;
                  if (timedIreq.has_start_time) {
                      latencies.push_back(std::chrono::duration_cast<Ms>(time_point - timedIreq.start).count());
                  }
                  ireq.set_callback(
                      [&ireq, time_point, &mutex, &finished_ireqs, &callback_exception, &cv](std::exception_ptr ex) {
                          // Keep callback small. This improves performance for fast (tens of thousands FPS) models
                          {
                              std::unique_lock<std::mutex> lock(mutex);
                              try {
                                  if (ex) {
                                      std::rethrow_exception(ex);
                                  }
                                  finished_ireqs.push_back({ireq, time_point, true});
                              } catch (const std::exception&) {
                                  if (!callback_exception) {
                                      callback_exception = std::current_exception();
                                  }
                              }
                          }
                          cv.notify_one();
                      });
                  ireq.start_async();
              }
          }
          auto end = std::chrono::steady_clock::now();
          double duration = std::chrono::duration_cast<Ms>(end - start).count();
          // Report results
          slog::info << "Count:      " << latencies.size() << " iterations" << slog::endl
                     << "Duration:   " << duration << " ms" << slog::endl
                     << "Latency:" << slog::endl;
          size_t percent = 50;
          LatencyMetrics{latencies, "", percent}.write_to_slog();
          slog::info << "Throughput: " << double_to_string(1000 * latencies.size() / duration) << " FPS" << slog::endl;
      } catch (const std::exception& ex) {
          slog::err << ex.what() << slog::endl;
          return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
  }
  ```

  </div>
  </TabItem>
</Tabs>

You can see the explicit description of each sample step at
[Integration Steps](../../../openvino-workflow/running-inference/index.md)
section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Running

<Tabs>
  <TabItem value="python" label="Python">
  ```console
  python throughput_benchmark.py <path_to_model> <device_name>(default: CPU)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  throughput_benchmark <path_to_model> <device_name>(default: CPU)
  ```
  </TabItem>
</Tabs>

To run the sample, you need to specify a model. You can get a model specific for
your inference task from one of model repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.

### Example

1. Download a pre-trained model.
2. You can convert it by using:
   
   <Tabs>
     <TabItem value="python" label="Python">
     ```python
     import openvino as ov

     ov_model = ov.convert_model('./models/googlenet-v1')
     # or, when model is a Python model object
     ov_model = ov.convert_model(googlenet-v1)
     ```
     </TabItem>
     <TabItem value="cli" label="CLI">
     ```console
     ovc ./models/googlenet-v1
     ```
     </TabItem>
   </Tabs>
3. Perform benchmarking, using the `googlenet-v1` model on a `CPU`:
   
   <Tabs>
     <TabItem value="python" label="Python">
     ```console
     python throughput_benchmark.py ./models/googlenet-v1.xml
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```console
     throughput_benchmark ./models/googlenet-v1.xml
     ```
     </TabItem>
   </Tabs>

## Sample Output

<Tabs>
  <TabItem value="python" label="Python">
  The application outputs performance results.

  ```console
  [ INFO ] OpenVINO:
  [ INFO ] Build ................................. <version>
  [ INFO ] Count:          2817 iterations
  [ INFO ] Duration:       10012.65 ms
  [ INFO ] Latency:
  [ INFO ]     Median:     13.80 ms
  [ INFO ]     Average:    14.10 ms
  [ INFO ]     Min:        8.35 ms
  [ INFO ]     Max:        28.38 ms
  [ INFO ] Throughput: 281.34 FPS
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  The application outputs performance results.

  ```console
  [ INFO ] OpenVINO:
  [ INFO ] Build ................................. <version>
  [ INFO ] Count:      1577 iterations
  [ INFO ] Duration:   15024.2 ms
  [ INFO ] Latency:
  [ INFO ]        Median:     38.02 ms
  [ INFO ]        Average:    38.08 ms
  [ INFO ]        Min:        25.23 ms
  [ INFO ]        Max:        49.16 ms
  [ INFO ] Throughput: 104.96 FPS
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Integrate the OpenVINO™ Runtime with Your Application](../../../openvino-workflow/running-inference/index.md)
- [Get Started with Samples](get-started-demos.md)
- [Using OpenVINO Samples](index.md)
- [Convert a Model](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md)
- [Throughput Benchmark Python Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/benchmark/throughput_benchmark/README.md)
- [Throughput Benchmark C++ Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/benchmark/throughput_benchmark/README.md)
