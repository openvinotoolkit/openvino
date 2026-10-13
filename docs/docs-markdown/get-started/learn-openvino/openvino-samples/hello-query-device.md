---
sidebar_position: 4
sidebar_label: "Hello Query Device Sample"
description: "Learn how to show metrics and default configuration values of inference devices using Query Device API feature (Python, C++)."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hello Query Device Sample

This sample demonstrates how to show OpenVINO™ Runtime devices and prints their
metrics and default configuration values using [Query Device API feature](../../../openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.md).
To build the sample, use instructions available at [Build the Sample Applications](get-started-demos.md#build-samples)
section in "Get Started with Samples" guide.

## How It Works

The sample queries all available OpenVINO™ Runtime devices and prints their
supported metrics and plugin configuration parameters.

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

  import openvino as ov


  def param_to_string(parameters) -> str:
      """Convert a list / tuple of parameters returned from OV to a string."""
      if isinstance(parameters, (list, tuple)):
          return ', '.join([str(x) for x in parameters])
      else:
          return str(parameters)


  def main():
      log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

      # --------------------------- Step 1. Initialize OpenVINO Runtime Core --------------------------------------------
      core = ov.Core()

      # --------------------------- Step 2. Get metrics of available devices --------------------------------------------
      log.info('Available devices:')
      for device in core.available_devices:
          log.info(f'{device} :')
          log.info('\tSUPPORTED_PROPERTIES:')
          for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
              if property_key not in ('SUPPORTED_PROPERTIES'):
                  try:
                      property_val = core.get_property(device, property_key)
                  except TypeError:
                      property_val = 'UNSUPPORTED TYPE'
                  log.info(f'\t\t{property_key}: {param_to_string(property_val)}')
          log.info('')

      # -----------------------------------------------------------------------------------------------------------------
      return 0


  if __name__ == '__main__':
      sys.exit(main())
  ```

  </div>
  </TabItem>
  <TabItem value="cpp" label="C++">
  <div style={{maxHeight: '400px', overflowY: 'auto'}}>

  ```cpp
  // Copyright (C) 2018-2026 Intel Corporation
  // SPDX-License-Identifier: Apache-2.0
  //

  #include <cstdlib>
  #include <iomanip>
  #include <memory>
  #include <set>
  #include <string>
  #include <tuple>
  #include <vector>

  // clang-format off
  #include "openvino/openvino.hpp"
  #include "samples/common.hpp"
  #include "samples/slog.hpp"
  // clang-format on

  /**
   * @brief Print OV Parameters
   * @param reference on OV Parameter
   * @return void
   */
  void print_any_value(const ov::Any& value) {
      if (value.empty()) {
          slog::info << "EMPTY VALUE" << slog::endl;
      } else {
          std::string stringValue = value.as<std::string>();
          slog::info << (stringValue.empty() ? "\"\"" : stringValue) << slog::endl;
      }
  }

  int main(int argc, char* argv[]) {
      try {
          // -------- Get OpenVINO runtime version --------
          slog::info << ov::get_openvino_version() << slog::endl;

          // -------- Parsing and validation of input arguments --------
          if (argc != 1) {
              std::cout << "Usage : " << argv[0] << std::endl;
              return EXIT_FAILURE;
          }

          // -------- Step 1. Initialize OpenVINO Runtime Core --------
          ov::Core core;

          // -------- Step 2. Get list of available devices --------
          std::vector<std::string> availableDevices = core.get_available_devices();

          // -------- Step 3. Query and print supported metrics and config keys --------
          slog::info << "Available devices: " << slog::endl;
          for (auto&& device : availableDevices) {
              slog::info << device << slog::endl;

              // Query supported properties and print all of them
              slog::info << "\tSUPPORTED_PROPERTIES: " << slog::endl;
              auto supported_properties = core.get_property(device, ov::supported_properties);
              for (auto&& property : supported_properties) {
                  if (property != ov::supported_properties.name()) {
                      slog::info << "\t\t" << (property.is_mutable() ? "Mutable: " : "Immutable: ") << property << " : "
                                 << slog::flush;
                      print_any_value(core.get_property(device, property));
                  }
              }

              slog::info << slog::endl;
          }
      } catch (const std::exception& ex) {
          std::cerr << std::endl << "Exception occurred: " << ex.what() << std::endl << std::flush;
          return EXIT_FAILURE;
      }

      return EXIT_SUCCESS;
  }
  ```

  </div>
  </TabItem>
</Tabs>

## Running

The sample has no command-line parameters. To see the report, run the following command:

<Tabs>
  <TabItem value="python" label="Python">
  ```console
  python hello_query_device.py
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  hello_query_device
  ```
  </TabItem>
</Tabs>

## Sample Output

The application prints all available devices with their supported metrics and
default values for configuration parameters.
For example:

<Tabs>
  <TabItem value="python" label="Python">
  ```console
  [ INFO ] Available devices:
  [ INFO ] CPU :
  [ INFO ]        SUPPORTED_PROPERTIES:
  [ INFO ]                AVAILABLE_DEVICES:
  [ INFO ]                FULL_DEVICE_NAME: Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
  [ INFO ]                OPTIMIZATION_CAPABILITIES: FP32, FP16, INT8, BIN
  [ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS: 1, 1, 1
  [ INFO ]                RANGE_FOR_STREAMS: 1, 8
  [ INFO ]                IMPORT_EXPORT_SUPPORT: True
  [ INFO ]                CACHE_DIR:
  [ INFO ]                ENABLE_CPU_PINNING: NO
  [ INFO ]                INFERENCE_NUM_THREADS: 0
  [ INFO ]                NUM_STREAMS: 1
  [ INFO ]                DUMP_EXEC_GRAPH_AS_DOT:
  [ INFO ]                INFERENCE_PRECISION_HINT: f32
  [ INFO ]                EXCLUSIVE_ASYNC_REQUESTS: NO
  [ INFO ]                PERFORMANCE_HINT:
  [ INFO ]                PERFORMANCE_HINT_NUM_REQUESTS: 0
  [ INFO ]                PERF_COUNT: NO
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  [ INFO ] OpenVINO Runtime version ......... <version>
  [ INFO ] Build ........... <build>
  [ INFO ]
  [ INFO ] Available devices:
  [ INFO ] CPU
  [ INFO ]        SUPPORTED_PROPERTIES:
  [ INFO ]                AVAILABLE_DEVICES : [  ]
  [ INFO ]                FULL_DEVICE_NAME : Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
  [ INFO ]                OPTIMIZATION_CAPABILITIES : [ FP32 FP16 INT8 BIN ]
  [ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS : { 1, 1, 1 }
  [ INFO ]                RANGE_FOR_STREAMS : { 1, 8 }
  [ INFO ]                IMPORT_EXPORT_SUPPORT : true
  [ INFO ]                CACHE_DIR : ""
  [ INFO ]                ENABLE_CPU_PINNING : NO
  [ INFO ]                INFERENCE_NUM_THREADS : 0
  [ INFO ]                NUM_STREAMS : 1
  [ INFO ]                DUMP_EXEC_GRAPH_AS_DOT : ""
  [ INFO ]                INFERENCE_PRECISION_HINT : f32
  [ INFO ]                EXCLUSIVE_ASYNC_REQUESTS : NO
  [ INFO ]                PERFORMANCE_HINT : ""
  [ INFO ]                PERFORMANCE_HINT_NUM_REQUESTS : 0
  [ INFO ]                PERF_COUNT : NO
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Integrate the OpenVINO™ Runtime with Your Application](../../../openvino-workflow/running-inference/index.md)
- [Get Started with Samples](get-started-demos.md)
- [Using OpenVINO™ Toolkit Samples](index.md)
- [Hello Query Device Python Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_query_device/README.md)
- [Hello Query Device C++ Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/hello_query_device/README.md)
