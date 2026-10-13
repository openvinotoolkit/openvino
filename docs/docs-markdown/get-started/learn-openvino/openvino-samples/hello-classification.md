---
sidebar_position: 2
sidebar_label: "Hello Classification Sample"
description: "Learn how to do inference of image classification models using Synchronous Inference Request API (Python, C++, C)."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hello Classification Sample

This sample demonstrates how to do inference of image classification models using
Synchronous Inference Request API. Before using the sample, refer to the following requirements:

- Models with only one input and output are supported.
- The sample accepts any file format supported by `core.read_model`.
- To build the sample, use instructions available at [Build the Sample Applications](get-started-demos.md#build-samples)
  section in "Get Started with Samples" guide.

## How It Works

At startup, the sample application sets log message capturing callback and reads command-line parameters.
Then it prepares input data, loads a specified model and image to the OpenVINO™ Runtime plugin, performs synchronous
inference, and processes output data, logging each step in a standard output stream.

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

  import cv2
  import numpy as np
  import openvino as ov


  def main():
      log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

      # Parsing and validation of input arguments
      if len(sys.argv) != 4:
          log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name>')
          return 1

      model_path = sys.argv[1]
      image_path = sys.argv[2]
      device_name = sys.argv[3]

  # --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
      log.info('Creating OpenVINO Runtime Core')
      core = ov.Core()

  # --------------------------- Step 2. Read a model --------------------------------------------------------------------
      log.info(f'Reading the model: {model_path}')
      # (.xml and .bin files) or (.onnx file)
      model = core.read_model(model_path)

      if len(model.inputs) != 1:
          log.error('Sample supports only single input topologies')
          return -1

      if len(model.outputs) != 1:
          log.error('Sample supports only single output topologies')
          return -1

  # --------------------------- Step 3. Set up input --------------------------------------------------------------------
      # Read input image
      image = cv2.imread(image_path)
      # Add N dimension
      input_tensor = np.expand_dims(image, 0)

  # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
      ppp = ov.preprocess.PrePostProcessor(model)

      _, h, w, _ = input_tensor.shape

      # 1) Set input tensor information:
      # - input() provides information about a single model input
      # - reuse precision and shape from already available `input_tensor`
      # - layout of data is 'NHWC'
      ppp.input().tensor() \
          .set_shape(input_tensor.shape) \
          .set_element_type(ov.Type.u8) \
          .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400

      # 2) Adding explicit preprocessing steps:
      # - apply linear resize from tensor spatial dims to model spatial dims
      ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)

      # 3) Here we suppose model has 'NCHW' layout for input
      ppp.input().model().set_layout(ov.Layout('NCHW'))

      # 4) Set output tensor information:
      # - precision of tensor is supposed to be 'f32'
      ppp.output().tensor().set_element_type(ov.Type.f32)

      # 5) Apply preprocessing modifying the original 'model'
      model = ppp.build()

  # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
      log.info('Loading the model to the plugin')
      compiled_model = core.compile_model(model, device_name)

  # --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
      log.info('Starting inference in synchronous mode')
      results = compiled_model.infer_new_request({0: input_tensor})

  # --------------------------- Step 7. Process output ------------------------------------------------------------------
      predictions = next(iter(results.values()))

      # Change a shape of a numpy.ndarray with results to get another one with one dimension
      probs = predictions.reshape(-1)

      # Get an array of 10 class IDs in descending order of probability
      top_10 = np.argsort(probs)[-10:][::-1]

      header = 'class_id probability'

      log.info(f'Image path: {image_path}')
      log.info('Top 10 results: ')
      log.info(header)
      log.info('-' * len(header))

      for class_id in top_10:
          probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
          log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}')

      log.info('')

  # ----------------------------------------------------------------------------------------------------------------------
      log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
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

  #include <iterator>
  #include <memory>
  #include <sstream>
  #include <string>
  #include <vector>

  // clang-format off
  #include "openvino/openvino.hpp"

  #include "samples/args_helper.hpp"
  #include "samples/common.hpp"
  #include "samples/classification_results.h"
  #include "samples/slog.hpp"
  #include "format_reader_ptr.h"
  // clang-format on

  /**
   * @brief Main with support Unicode paths, wide strings
   */
  int tmain(int argc, tchar* argv[]) {
      try {
          // -------- Set OpenVINO log message capturing callback --------
          const std::function<void(std::string_view)> log_callback{[](std::string_view msg) {
              slog::info << msg;
          }};
          ov::util::set_log_callback(log_callback);

          // -------- Get OpenVINO runtime version --------
          slog::info << ov::get_openvino_version() << slog::endl;

          // -------- Parsing and validation of input arguments --------
          if (argc != 4) {
              slog::info << "Usage : " << TSTRING2STRING(argv[0]) << " <path_to_model> <path_to_image> <device_name>"
                         << slog::endl;
              return EXIT_FAILURE;
          }

          const std::string args = TSTRING2STRING(argv[0]);
          const std::string model_path = TSTRING2STRING(argv[1]);
          const std::string image_path = TSTRING2STRING(argv[2]);
          const std::string device_name = TSTRING2STRING(argv[3]);

          // -------- Step 1. Initialize OpenVINO Runtime Core --------
          ov::Core core;

          // -------- Step 2. Read a model --------
          slog::info << "Loading model files: " << model_path << slog::endl;
          std::shared_ptr<ov::Model> model = core.read_model(model_path);
          printInputAndOutputsInfo(*model);

          OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
          OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

          // -------- Step 3. Set up input

          // Read input image to a tensor and set it to an infer request
          // without resize and layout conversions
          FormatReader::ReaderPtr reader(image_path.c_str());
          if (reader.get() == nullptr) {
              std::stringstream ss;
              ss << "Image " + image_path + " cannot be read!";
              throw std::logic_error(ss.str());
          }

          ov::element::Type input_type = ov::element::u8;
          ov::Shape input_shape = {1, reader->height(), reader->width(), 3};
          std::shared_ptr<unsigned char> input_data = reader->getData();

          // just wrap image data by ov::Tensor without allocating of new memory
          ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());

          const ov::Layout tensor_layout{"NHWC"};

          // -------- Step 4. Configure preprocessing --------

          ov::preprocess::PrePostProcessor ppp(model);

          // 1) Set input tensor information:
          // - input() provides information about a single model input
          // - reuse precision and shape from already available `input_tensor`
          // - layout of data is 'NHWC'
          ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
          // 2) Adding explicit preprocessing steps:
          // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
          // - apply linear resize from tensor spatial dims to model spatial dims
          ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
          // 4) Suppose model has 'NCHW' layout for input
          ppp.input().model().set_layout("NCHW");
          // 5) Set output tensor information:
          // - precision of tensor is supposed to be 'f32'
          ppp.output().tensor().set_element_type(ov::element::f32);

          // 6) Apply preprocessing modifying the original 'model'
          model = ppp.build();

          // -------- Step 5. Loading a model to the device --------
          ov::CompiledModel compiled_model = core.compile_model(model, device_name);

          // -------- Step 6. Create an infer request --------
          ov::InferRequest infer_request = compiled_model.create_infer_request();
          // -----------------------------------------------------------------------------------------------------

          // -------- Step 7. Prepare input --------
          infer_request.set_input_tensor(input_tensor);

          // -------- Step 8. Do inference synchronously --------
          infer_request.infer();

          // -------- Step 9. Process output
          const ov::Tensor& output_tensor = infer_request.get_output_tensor();

          // Print classification results
          ClassificationResult classification_result(output_tensor, {image_path});
          classification_result.show();
          // -----------------------------------------------------------------------------------------------------
      } catch (const std::exception& ex) {
          std::cerr << ex.what() << std::endl;
          return EXIT_FAILURE;
      }

      return EXIT_SUCCESS;
  }
  ```

  </div>
  </TabItem>
  <TabItem value="c" label="C">
  <div style={{maxHeight: '400px', overflowY: 'auto'}}>

  ```c
  // Copyright (C) 2018-2026 Intel Corporation
  // SPDX-License-Identifier: Apache-2.0
  //

  #include <stdbool.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>

  #include "samples_util/path_util.h"
  // Uses windows.h must be before openvino/c/openvino.h
  #include "infer_result_util.h"
  #include "opencv_c_wrapper.h"
  #include "openvino/c/openvino.h"

  void print_model_input_output_info(ov_model_t* model) {
      char* friendly_name = NULL;
      ov_model_get_friendly_name(model, &friendly_name);
      printf("[INFO] model name: %s \n", friendly_name);
      ov_free(friendly_name);
  }

  #define CHECK_STATUS(return_status)                                                      \
      if (return_status != OK) {                                                           \
          fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
          goto err;                                                                        \
      }

  int main(int argc, char** argv) {
      // -------- Check input parameters --------
      if (argc != 4) {
          printf("Usage : ./hello_classification_c <path_to_model> <path_to_image> "
                 "<device_name>\n");
          return EXIT_FAILURE;
      }

      ov_core_t* core = NULL;
      ov_model_t* model = NULL;
      ov_tensor_t* tensor = NULL;
      ov_preprocess_prepostprocessor_t* preprocess = NULL;
      ov_preprocess_input_info_t* input_info = NULL;
      ov_model_t* new_model = NULL;
      ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
      ov_preprocess_preprocess_steps_t* input_process = NULL;
      ov_preprocess_input_model_info_t* p_input_model = NULL;
      ov_preprocess_output_info_t* output_info = NULL;
      ov_preprocess_output_tensor_info_t* output_tensor_info = NULL;
      ov_compiled_model_t* compiled_model = NULL;
      ov_infer_request_t* infer_request = NULL;
      ov_tensor_t* output_tensor = NULL;
      struct infer_result* results = NULL;
      ov_layout_t* input_layout = NULL;
      ov_layout_t* model_layout = NULL;
      ov_shape_t input_shape = {.rank = 0, .dims = NULL};
      ov_output_const_port_t* output_port = NULL;
      ov_output_const_port_t* input_port = NULL;
      c_mat_t img = {0};

      // -------- Get OpenVINO runtime version --------
      ov_version_t version;
      CHECK_STATUS(ov_get_openvino_version(&version));
      printf("---- OpenVINO INFO----\n");
      printf("Description : %s \n", version.description);
      printf("Build number: %s \n", version.buildNumber);
      ov_version_free(&version);

      // -------- Parsing and validation of input arguments --------
      const char* input_model = argv[1];
      char input_image_path[PATH_MAX];
      if (!sanitize_path(argv[2], input_image_path, sizeof(input_image_path))) {
          goto err;
      }
      const char* device_name = argv[3];

      // -------- Step 1. Initialize OpenVINO Runtime Core --------
      CHECK_STATUS(ov_core_create(&core));

      // -------- Step 2. Read a model --------
      printf("[INFO] Loading model files: %s\n", input_model);
      CHECK_STATUS(ov_core_read_model(core, input_model, NULL, &model));
      print_model_input_output_info(model);

      CHECK_STATUS(ov_model_const_output(model, &output_port));
      if (!output_port) {
          fprintf(stderr, "[ERROR] Sample supports models with 1 output only %d\n", __LINE__);
          goto err;
      }

      CHECK_STATUS(ov_model_const_input(model, &input_port));
      if (!input_port) {
          fprintf(stderr, "[ERROR] Sample supports models with 1 input only %d\n", __LINE__);
          goto err;
      }

      // -------- Step 3. Set up input

      image_read(input_image_path, &img);
      ov_element_type_e input_type = U8;
      int64_t dims[4] = {1, (size_t)img.mat_height, (size_t)img.mat_width, 3};
      ov_shape_create(4, dims, &input_shape);
      CHECK_STATUS(ov_tensor_create_from_host_ptr(input_type, input_shape, img.mat_data, &tensor));

      // -------- Step 4. Configure preprocessing --------
      CHECK_STATUS(ov_preprocess_prepostprocessor_create(model, &preprocess));
      CHECK_STATUS(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));

      CHECK_STATUS(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
      CHECK_STATUS(ov_preprocess_input_tensor_info_set_from(input_tensor_info, tensor));

      const char* input_layout_desc = "NHWC";
      CHECK_STATUS(ov_layout_create(input_layout_desc, &input_layout));
      CHECK_STATUS(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, input_layout));

      CHECK_STATUS(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
      CHECK_STATUS(ov_preprocess_preprocess_steps_resize(input_process, RESIZE_LINEAR));
      CHECK_STATUS(ov_preprocess_input_info_get_model_info(input_info, &p_input_model));

      const char* model_layout_desc = "NCHW";
      CHECK_STATUS(ov_layout_create(model_layout_desc, &model_layout));
      CHECK_STATUS(ov_preprocess_input_model_info_set_layout(p_input_model, model_layout));

      CHECK_STATUS(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
      CHECK_STATUS(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
      CHECK_STATUS(ov_preprocess_output_set_element_type(output_tensor_info, F32));

      CHECK_STATUS(ov_preprocess_prepostprocessor_build(preprocess, &new_model));

      // -------- Step 5. Loading a model to the device --------
      CHECK_STATUS(ov_core_compile_model(core, new_model, device_name, 0, &compiled_model));

      // -------- Step 6. Create an infer request --------
      CHECK_STATUS(ov_compiled_model_create_infer_request(compiled_model, &infer_request));

      // -------- Step 7. Prepare input --------
      CHECK_STATUS(ov_infer_request_set_input_tensor_by_index(infer_request, 0, tensor));

      // -------- Step 8. Do inference synchronously --------
      CHECK_STATUS(ov_infer_request_infer(infer_request));

      // -------- Step 9. Process output
      CHECK_STATUS(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
      // Print classification results
      size_t results_num;
      results = tensor_to_infer_result(output_tensor, &results_num);
      infer_result_sort(results, results_num);
      size_t top = 10;
      if (top > results_num) {
          top = results_num;
      }
      printf("\nTop %zu results:\n", top);
      print_infer_result(results, top, input_image_path);

      // -------- free allocated resources --------
  err:
      free(results);
      image_free(&img);
      ov_shape_free(&input_shape);
      ov_output_const_port_free(output_port);
      ov_output_const_port_free(input_port);
      if (output_tensor)
          ov_tensor_free(output_tensor);
      if (infer_request)
          ov_infer_request_free(infer_request);
      if (compiled_model)
          ov_compiled_model_free(compiled_model);
      if (input_layout)
          ov_layout_free(input_layout);
      if (model_layout)
          ov_layout_free(model_layout);
      if (output_tensor_info)
          ov_preprocess_output_tensor_info_free(output_tensor_info);
      if (output_info)
          ov_preprocess_output_info_free(output_info);
      if (p_input_model)
          ov_preprocess_input_model_info_free(p_input_model);
      if (input_process)
          ov_preprocess_preprocess_steps_free(input_process);
      if (input_tensor_info)
          ov_preprocess_input_tensor_info_free(input_tensor_info);
      if (input_info)
          ov_preprocess_input_info_free(input_info);
      if (preprocess)
          ov_preprocess_prepostprocessor_free(preprocess);
      if (new_model)
          ov_model_free(new_model);
      if (tensor)
          ov_tensor_free(tensor);
      if (model)
          ov_model_free(model);
      if (core)
          ov_core_free(core);
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
  python hello_classification.py <path_to_model> <path_to_image> <device_name>
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  hello_classification <path_to_model> <path_to_image> <device_name>
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```console
  hello_classification_c <path_to_model> <path_to_image> <device_name>
  ```
  </TabItem>
</Tabs>

To run the sample, you need to specify a model and an image:

- You can get a model specific for your inference task from one of model
  repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.
- You can use images from the media files collection available at
  [the storage](https://storage.openvinotoolkit.org/data/test_data).

:::note
- By default, OpenVINO™ Toolkit Samples and demos expect input with BGR
  channels order. If you trained your model to work with RGB order, you need
  to manually rearrange the default channels order in the sample or demo
  application or reconvert your model using model conversion API with
  `reverse_input_channels` argument specified. For more information about
  the argument, refer to the **Color Conversion** section of
  [Preprocessing API](../../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details/index.md).
- Before running the sample with a trained model, make sure the model is
  converted to the intermediate representation (IR) format (\*.xml + \*.bin)
  using the [model conversion API](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md).
- The sample accepts models in ONNX format (.onnx) that do not require preprocessing.
- The sample supports NCHW model layout only.
:::

### Example

1. Download a pre-trained model.
2. You can convert it by using:
   
   <Tabs>
     <TabItem value="python" label="Python">
     ```python
     import openvino as ov

     ov_model = ov.convert_model('./models/alexnet')
     # or, when model is a Python model object
     ov_model = ov.convert_model(alexnet)
     ```
     </TabItem>
     <TabItem value="cli" label="CLI">
     ```console
     ovc ./models/alexnet
     ```
     </TabItem>
   </Tabs>
3. Perform inference of an image, using a model on a `GPU`, for example:
   
   <Tabs>
     <TabItem value="python" label="Python">
     ```console
     python hello_classification.py ./models/alexnet/alexnet.xml ./images/banana.jpg GPU
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```console
     hello_classification ./models/googlenet-v1.xml ./images/car.bmp GPU
     ```
     </TabItem>
     <TabItem value="c" label="C">
     ```console
     hello_classification_c alexnet.xml ./opt/intel/openvino/samples/scripts/car.png GPU
     ```
     </TabItem>
   </Tabs>

## Sample Output

<Tabs>
  <TabItem value="python" label="Python">
  The sample application logs each step in a standard output stream and
  outputs top-10 inference results.

  ```console
  [ INFO ] Creating OpenVINO Runtime Core
  [ INFO ] Reading the model: /models/alexnet/alexnet.xml
  [ INFO ] Loading the model to the plugin
  [ INFO ] Starting inference in synchronous mode
  [ INFO ] Image path: /images/banana.jpg
  [ INFO ] Top 10 results:
  [ INFO ] class_id probability
  [ INFO ] --------------------
  [ INFO ] 954      0.9703885
  [ INFO ] 666      0.0219518
  [ INFO ] 659      0.0033120
  [ INFO ] 435      0.0008246
  [ INFO ] 809      0.0004433
  [ INFO ] 502      0.0003852
  [ INFO ] 618      0.0002906
  [ INFO ] 910      0.0002848
  [ INFO ] 951      0.0002427
  [ INFO ] 961      0.0002213
  [ INFO ]
  [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  The application outputs top-10 inference results.

  ```console
  [ INFO ] OpenVINO Runtime version ......... <version>
  [ INFO ] Build ........... <build>
  [ INFO ]
  [ INFO ] Loading model files: /models/googlenet-v1.xml
  [ INFO ] model name: GoogleNet
  [ INFO ]     inputs
  [ INFO ]         input name: data
  [ INFO ]         input type: f32
  [ INFO ]         input shape: {1, 3, 224, 224}
  [ INFO ]     outputs
  [ INFO ]         output name: prob
  [ INFO ]         output type: f32
  [ INFO ]         output shape: {1, 1000}

  Top 10 results:

  Image /images/car.bmp

  classid probability
  ------- -----------
  656     0.8139648
  654     0.0550537
  468     0.0178375
  436     0.0165405
  705     0.0111694
  817     0.0105820
  581     0.0086823
  575     0.0077515
  734     0.0064468
  785     0.0043983
  ```
  </TabItem>
  <TabItem value="c" label="C">
  The application outputs top-10 inference results.

  ```console
  Top 10 results:

  Image /opt/intel/openvino/samples/scripts/car.png

  classid probability
  ------- -----------
  656       0.666479
  654       0.112940
  581       0.068487
  874       0.033385
  436       0.026132
  817       0.016731
  675       0.010980
  511       0.010592
  569       0.008178
  717       0.006336

  This sample is an API example, for any performance measurements use the dedicated benchmark_app tool.
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Integrate the OpenVINO™ Runtime with Your Application](../../../openvino-workflow/running-inference/index.md)
- [Get Started with Samples](get-started-demos.md)
- [Using OpenVINO Samples](index.md)
- [Convert a Model](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md)
- [OpenVINO Runtime C API](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__c__api.html)
- [Hello Classification Python Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_classification/README.md)
- [Hello Classification C++ Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/hello_classification/README.md)
- [Hello Classification C Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/c/hello_classification/README.md)
