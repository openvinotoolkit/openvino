---
sidebar_position: 8
sidebar_label: "Sync Benchmark Sample"
description: "Learn how to estimate performance of a model using Synchronous Inference Request API (Python, C++)."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sync Benchmark Sample

This sample demonstrates how to estimate performance of a model using Synchronous
Inference Request API. It makes sense to use synchronous inference only in latency
oriented scenarios. Models with static input shapes are supported.
This sample does not have other configurable command-line
arguments. Feel free to modify sample's source code to try out different options.
Before using the sample, refer to the following requirements:

- The sample accepts any file format supported by `core.read_model`.
- The sample has been validated with: the yolo-v3-tf and face-detection-0200 models.
- To build the sample, use instructions available at [Build the Sample Applications](get-started-demos.md#build-samples)
  section in "Get Started with Samples" guide.

## How It Works

The sample compiles a model for a given device, randomly generates input data,
performs synchronous inference multiple times for a given number of seconds.
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
  import statistics
  import sys
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
      # Optimize for latency. Most of the devices are configured for latency by default,
      # but there are exceptions like GNA
      latency = {'PERFORMANCE_HINT': 'LATENCY'}

      # Create Core and use it to compile a model.
      # Select the device by providing the name as the second parameter to CLI.
      # Using MULTI device is pointless in sync scenario
      # because only one instance of openvino.InferRequest is used
      core = ov.Core()
      compiled_model = core.compile_model(sys.argv[1], device_name, latency)
      ireq = compiled_model.create_infer_request()
      # Fill input data for the ireq
      for model_input in compiled_model.inputs:
          fill_tensor_random(ireq.get_tensor(model_input))
      # Warm up
      ireq.infer()
      # Benchmark for seconds_to_run seconds and at least niter iterations
      seconds_to_run = 10
      niter = 10
      latencies = []
      start = perf_counter()
      time_point = start
      time_point_to_finish = start + seconds_to_run
      while time_point < time_point_to_finish or len(latencies) < niter:
          ireq.infer()
          iter_end = perf_counter()
          latencies.append((iter_end - time_point) * 1e3)
          time_point = iter_end
      end = time_point
      duration = end - start
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
          // Optimize for latency. Most of the devices are configured for latency by default,
          // but there are exceptions like GNA
          ov::AnyMap latency{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}};

          // Create ov::Core and use it to compile a model.
          // Select the device by providing the name as the second parameter to CLI.
          // Using MULTI device is pointless in sync scenario
          // because only one instance of ov::InferRequest is used
          ov::Core core;
          ov::CompiledModel compiled_model = core.compile_model(argv[1], device_name, latency);
          ov::InferRequest ireq = compiled_model.create_infer_request();
          // Fill input data for the ireq
          for (const ov::Output<const ov::Node>& model_input : compiled_model.inputs()) {
              fill_tensor_random(ireq.get_tensor(model_input));
          }
          // Warm up
          ireq.infer();
          // Benchmark for seconds_to_run seconds and at least niter iterations
          std::chrono::seconds seconds_to_run{10};
          size_t niter = 10;
          std::vector<double> latencies;
          latencies.reserve(niter);
          auto start = std::chrono::steady_clock::now();
          auto time_point = start;
          auto time_point_to_finish = start + seconds_to_run;
          while (time_point < time_point_to_finish || latencies.size() < niter) {
              ireq.infer();
              auto iter_end = std::chrono::steady_clock::now();
              latencies.push_back(std::chrono::duration_cast<Ms>(iter_end - time_point).count());
              time_point = iter_end;
          }
          auto end = time_point;
          double duration = std::chrono::duration_cast<Ms>(end - start).count();
          // Report results
          slog::info << "Count:      " << latencies.size() << " iterations" << slog::endl
                     << "Duration:   " << duration << " ms" << slog::endl
                     << "Latency:" << slog::endl;
          size_t percent = 50;
          LatencyMetrics{latencies, "", percent}.write_to_slog();
          slog::info << "Throughput: " << double_to_string(latencies.size() * 1000 / duration) << " FPS" << slog::endl;
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

You can see the explicit description of
each sample step at [Integration Steps](../../../openvino-workflow/running-inference/index.md)
section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Running

<Tabs>
  <TabItem value="python" label="Python">
  ```console
  python sync_benchmark.py <path_to_model> <device_name>(default: CPU)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  sync_benchmark <path_to_model> <device_name>(default: CPU)
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
     python sync_benchmark.py googlenet-v1.xml
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```console
     sync_benchmark googlenet-v1.xml
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
  [ INFO ] Count:          2333 iterations
  [ INFO ] Duration:       10003.59 ms
  [ INFO ] Latency:
  [ INFO ]     Median:     3.90 ms
  [ INFO ]     Average:    4.29 ms
  [ INFO ]     Min:        3.30 ms
  [ INFO ]     Max:        10.11 ms
  [ INFO ] Throughput: 233.22 FPS
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  The application outputs performance results.

  ```console
  [ INFO ] OpenVINO:
  [ INFO ] Build ................................. <version>
  [ INFO ] Count:      992 iterations
  [ INFO ] Duration:   15009.8 ms
  [ INFO ] Latency:
  [ INFO ]        Median:     14.00 ms
  [ INFO ]        Average:    15.13 ms
  [ INFO ]        Min:        9.33 ms
  [ INFO ]        Max:        53.60 ms
  [ INFO ] Throughput: 66.09 FPS
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Integrate the OpenVINO™ Runtime with Your Application](../../../openvino-workflow/running-inference/index.md)
- [Get Started with Samples](get-started-demos.md)
- [Using OpenVINO Samples](index.md)
- [Convert a Model](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md)
- [Sync Benchmark Python Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/benchmark/sync_benchmark/README.md)
- [Sync Benchmark C++ Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/benchmark/sync_benchmark/README.md)
