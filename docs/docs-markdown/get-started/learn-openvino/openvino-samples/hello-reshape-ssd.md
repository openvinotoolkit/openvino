---
sidebar_position: 5
sidebar_label: "Hello Reshape SSD Sample"
description: "Learn how to do inference of object detection models using shape inference feature and Synchronous Inference Request API (Python, C++)."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hello Reshape SSD Sample

This sample demonstrates how to do synchronous inference of object detection models
using [Shape Inference feature](../../../openvino-workflow/running-inference/model-input-output/changing-input-shape.md). Before
using the sample, refer to the following requirements:

- Models with only one input and output are supported.
- The sample accepts any file format supported by `core.read_model`.
- The sample has been validated with the person-detection-retail-0013
  model and the NCHW layout format.
- To build the sample, use instructions available at [Build the Sample Applications](get-started-demos.md#build-samples)
  section in "Get Started with Samples" guide.

## How It Works

At startup, the sample application reads command-line parameters, prepares input data, loads a specified model and image to the OpenVINO™ Runtime plugin, performs synchronous inference, and processes output data.
As a result, the program creates an output image, logging each step in a standard output stream.

<Tabs>
  <TabItem value="python" label="Python">
  <div style={{maxHeight: '400px', overflowY: 'auto'}}>

  ```python
  #!/usr/bin/env python3
  # -*- coding: utf-8 -*-
  # Copyright (C) 2018-2026 Intel Corporation
  # SPDX-License-Identifier: Apache-2.0

  import logging as log
  import os
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

      log.info('Reshaping the model to the height and width of the input image')
      n, h, w, c = input_tensor.shape
      model.reshape({model.input().get_any_name(): ov.PartialShape((n, c, h, w))})

  # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
      ppp = ov.preprocess.PrePostProcessor(model)

      # 1) Set input tensor information:
      # - input() provides information about a single model input
      # - precision of tensor is supposed to be 'u8'
      # - layout of data is 'NHWC'
      ppp.input().tensor() \
          .set_element_type(ov.Type.u8) \
          .set_layout(ov.Layout('NHWC'))  # noqa: N400

      # 2) Here we suppose model has 'NCHW' layout for input
      ppp.input().model().set_layout(ov.Layout('NCHW'))

      # 3) Set output tensor information:
      # - precision of tensor is supposed to be 'f32'
      ppp.output().tensor().set_element_type(ov.Type.f32)

      # 4) Apply preprocessing modifying the original 'model'
      model = ppp.build()

  # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
      log.info('Loading the model to the plugin')
      compiled_model = core.compile_model(model, device_name)

  # --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
      log.info('Starting inference in synchronous mode')
      results = compiled_model.infer_new_request({0: input_tensor})

  # ---------------------------Step 6. Process output--------------------------------------------------------------------
      predictions = next(iter(results.values()))

      # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
      # where N is the number of detected bounding boxes
      detections = predictions.reshape(-1, 7)

      for detection in detections:
          confidence = detection[2]

          if confidence > 0.5:
              class_id = int(detection[1])

              xmin = int(detection[3] * w)
              ymin = int(detection[4] * h)
              xmax = int(detection[5] * w)
              ymax = int(detection[6] * h)

              log.info(f'Found: class_id = {class_id}, confidence = {confidence:.2f}, ' f'coords = ({xmin}, {ymin}), ({xmax}, {ymax})')

              # Draw a bounding box on a output image
              cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

      cv2.imwrite('out.bmp', image)

      if os.path.exists('out.bmp'):
          log.info('Image out.bmp was created!')
      else:
          log.error('Image out.bmp was not created. Check your permissions.')

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

  #include <memory>
  #include <string>
  #include <vector>

  // clang-format off
  #include "openvino/openvino.hpp"
  #include "openvino/opsets/opset9.hpp"

  #include "format_reader_ptr.h"
  #include "samples/args_helper.hpp"
  #include "samples/common.hpp"
  #include "samples/slog.hpp"
  // clang-format on

  // thickness of a line (in pixels) to be used for bounding boxes
  constexpr int BBOX_THICKNESS = 2;

  using namespace ov::preprocess;

  int main(int argc, char* argv[]) {
      try {
          // -------- Get OpenVINO runtime version -----------------------------
          slog::info << ov::get_openvino_version() << slog::endl;

          // --------------------------- Parsing and validation of input arguments
          if (argc != 4) {
              std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device>" << std::endl;
              return EXIT_FAILURE;
          }
          const std::string model_path{argv[1]};
          const std::string image_path{argv[2]};
          const std::string device_name{argv[3]};
          // -------------------------------------------------------------------

          // Step 1. Initialize OpenVINO Runtime core
          ov::Core core;
          // -------------------------------------------------------------------

          // Step 2. Read a model
          slog::info << "Loading model files: " << model_path << slog::endl;
          std::shared_ptr<ov::Model> model = core.read_model(model_path);
          printInputAndOutputsInfo(*model);

          // Step 3. Validate model inputs and outputs
          OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
          OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

          // SSD has an additional post-processing DetectionOutput layer that simplifies output filtering,
          // try to find it.
          const ov::NodeVector ops = model->get_ops();
          const auto it = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& node) {
              return std::string{node->get_type_name()} ==
                     std::string{ov::opset9::DetectionOutput::get_type_info_static().name};
          });
          if (it == ops.end()) {
              throw std::logic_error("model does not contain DetectionOutput layer");
          }
          // -------------------------------------------------------------------

          // Step 4. Read input image

          // Read input image without resize
          FormatReader::ReaderPtr reader(image_path.c_str());
          if (reader.get() == nullptr) {
              std::cout << "Image " + image_path + " cannot be read!" << std::endl;
              return 1;
          }

          std::shared_ptr<unsigned char> image_data = reader->getData();
          size_t image_channels = 3;
          size_t image_width = reader->width();
          size_t image_height = reader->height();
          // -------------------------------------------------------------------

          // Step 5. Reshape model to image size and batch size
          // assume model layout NCHW
          const ov::Layout model_layout{"NCHW"};

          ov::Shape tensor_shape = model->input().get_shape();

          size_t batch_size = 1;

          tensor_shape[ov::layout::batch_idx(model_layout)] = batch_size;
          tensor_shape[ov::layout::channels_idx(model_layout)] = image_channels;
          tensor_shape[ov::layout::height_idx(model_layout)] = image_height;
          tensor_shape[ov::layout::width_idx(model_layout)] = image_width;

          std::cout << "Reshape network to the image size = [" << image_height << "x" << image_width << "] " << std::endl;
          model->reshape({{model->input().get_any_name(), tensor_shape}});
          printInputAndOutputsInfo(*model);
          // -------------------------------------------------------------------

          // Step 6. Configure model preprocessing
          const ov::Layout tensor_layout{"NHWC"};

          // clang-format off
          ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

          // 1) input() with no args assumes a model has a single input
          ov::preprocess::InputInfo& input_info = ppp.input();
          // 2) Set input tensor information:
          // - precision of tensor is supposed to be 'u8'
          // - layout of data is 'NHWC'
          input_info.tensor().
                set_element_type(ov::element::u8).
                set_layout(tensor_layout);
          // 3) Adding explicit preprocessing steps:
          // - convert u8 to f32
          // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
          ppp.input().preprocess().
              convert_element_type(ov::element::f32).
              convert_layout("NCHW");
          // 4) Here we suppose model has 'NCHW' layout for input
          input_info.model().set_layout("NCHW");
          // 5) output () with no args assumes a model has a single output
          ov::preprocess::OutputInfo& output_info = ppp.output();
          // 6) declare output element type as FP32
          output_info.tensor().set_element_type(ov::element::f32);

          // 7) Apply preprocessing modifying the original 'model'
          model = ppp.build();
          // clang-format on
          // -------------------------------------------------------------------

          // Step 7. Loading a model to the device
          ov::CompiledModel compiled_model = core.compile_model(model, device_name);
          // -------------------------------------------------------------------

          // Step 8. Create an infer request
          ov::InferRequest infer_request = compiled_model.create_infer_request();

          // Step 9. Fill model with input data
          ov::Tensor input_tensor = infer_request.get_input_tensor();

          // copy NHWC data from image to tensor with batch
          unsigned char* image_data_ptr = image_data.get();
          unsigned char* tensor_data_ptr = input_tensor.data<unsigned char>();
          size_t image_size = image_width * image_height * image_channels;
          for (size_t i = 0; i < image_size; i++) {
              tensor_data_ptr[i] = image_data_ptr[i];
          }
          // -------------------------------------------------------------------

          // Step 10. Do inference synchronously
          infer_request.infer();

          // Step 11. Get output data from the model
          ov::Tensor output_tensor = infer_request.get_output_tensor();

          ov::Shape output_shape = model->output().get_shape();
          const size_t ssd_object_count = output_shape[2];
          const size_t ssd_object_size = output_shape[3];

          const float* detections = output_tensor.data<const float>();
          // -------------------------------------------------------------------

          std::vector<int> boxes;
          std::vector<int> classes;

          // Step 12. Parse SSD output
          for (size_t object = 0; object < ssd_object_count; object++) {
              int image_id = static_cast<int>(detections[object * ssd_object_size + 0]);
              if (image_id < 0) {
                  break;
              }

              // detection, has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
              int label = static_cast<int>(detections[object * ssd_object_size + 1]);
              float confidence = detections[object * ssd_object_size + 2];
              int xmin = static_cast<int>(detections[object * ssd_object_size + 3] * image_width);
              int ymin = static_cast<int>(detections[object * ssd_object_size + 4] * image_height);
              int xmax = static_cast<int>(detections[object * ssd_object_size + 5] * image_width);
              int ymax = static_cast<int>(detections[object * ssd_object_size + 6] * image_height);

              if (confidence > 0.5f) {
                  // collect only objects with >50% probability
                  classes.push_back(label);
                  boxes.push_back(xmin);
                  boxes.push_back(ymin);
                  boxes.push_back(xmax - xmin);
                  boxes.push_back(ymax - ymin);

                  std::cout << "[" << object << "," << label << "] element, prob = " << confidence << ",    (" << xmin
                            << "," << ymin << ")-(" << xmax << "," << ymax << ")" << std::endl;
              }
          }

          // draw bounding boxes on the image
          addRectangles(image_data.get(), image_height, image_width, boxes, classes, BBOX_THICKNESS);

          const std::string image_name = "hello_reshape_ssd_output.bmp";
          if (writeOutputBmp(image_name, image_data.get(), image_height, image_width)) {
              std::cout << "The resulting image was saved in the file: " + image_name << std::endl;
          } else {
              throw std::logic_error(std::string("Can't create a file: ") + image_name);
          }

      } catch (const std::exception& ex) {
          std::cerr << ex.what() << std::endl;
          return EXIT_FAILURE;
      }
      std::cout << std::endl
                << "This sample is an API example, for any performance measurements "
                   "please use the dedicated benchmark_app tool"
                << std::endl;
      return EXIT_SUCCESS;
  }
  ```

  </div>
  </TabItem>
</Tabs>

You can see the explicit description of
each sample step at [Integration Steps](../../../openvino-workflow/running-inference/index.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Running

<Tabs>
  <TabItem value="python" label="Python">
  ```console
  python hello_reshape_ssd.py <path_to_model> <path_to_image> <device_name>
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  hello_reshape_ssd <path_to_model> <path_to_image> <device_name>
  ```
  </TabItem>
</Tabs>

To run the sample, you need to specify a model and an image:

- You can get a model specific for your inference task from one of model
  repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.
- You can use images from the media files collection available at
  [the storage](https://storage.openvinotoolkit.org/data/test_data).

:::note
- By default, OpenVINO™ Toolkit Samples and demos expect input with BGR channels
  order. If you trained your model to work with RGB order, you need to manually
  rearrange the default channels order in the sample or demo application or
  reconvert your model using model conversion API with `reverse_input_channels`
  argument specified. For more information about the argument, refer to the
  **Color Conversion** section of
  [Preprocessing API](../../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details/index.md).
- Before running the sample with a trained model, make sure the model is
  converted to the intermediate representation (IR) format (\*.xml + \*.bin)
  using [model conversion API](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md).
- The sample accepts models in ONNX format (.onnx) that do not require preprocessing.
:::

### Example

1. Download a pre-trained model:
2. You can convert it by using:
   
   <Tabs>
     <TabItem value="python" label="Python">
     ```python
     import openvino as ov

     ov_model = ov.convert_model('./test_data/models/mobilenet-ssd')
     # or, when model is a Python model object
     ov_model = ov.convert_model(mobilenet-ssd)
     ```
     </TabItem>
     <TabItem value="cli" label="CLI">
     ```console
     ovc ./test_data/models/mobilenet-ssd
     ```
     </TabItem>
   </Tabs>

4. Perform inference of an image, using a model on a `GPU`, for example:
   
   <Tabs>
     <TabItem value="python" label="Python">
     ```console
     python hello_reshape_ssd.py ./test_data/models/mobilenet-ssd.xml banana.jpg GPU
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```console
     hello_reshape_ssd ./models/person-detection-retail-0013.xml person_detection.bmp GPU
     ```
     </TabItem>
   </Tabs>

## Sample Output

<Tabs>
  <TabItem value="python" label="Python">
  The sample application logs each step in a standard output stream and
  creates an output image, drawing bounding boxes for inference results
  with an over 50% confidence.

  ```console
  [ INFO ] Creating OpenVINO Runtime Core
  [ INFO ] Reading the model: C:/test_data/models/mobilenet-ssd.xml
  [ INFO ] Reshaping the model to the height and width of the input image
  [ INFO ] Loading the model to the plugin
  [ INFO ] Starting inference in synchronous mode
  [ INFO ] Found: class_id = 52, confidence = 0.98, coords = (21, 98), (276, 210)
  [ INFO ] Image out.bmp was created!
  [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  The application renders an image with detected objects enclosed in rectangles.
  It outputs the list of classes of the detected objects along with the
  respective confidence values and the coordinates of the rectangles to the
  standard output stream.

  ```console
  [ INFO ] OpenVINO Runtime version ......... <version>
  [ INFO ] Build ........... <build>
  [ INFO ]
  [ INFO ] Loading model files: \models\person-detection-retail-0013.xml
  [ INFO ] model name: ResMobNet_v4 (LReLU) with single SSD head
  [ INFO ]     inputs
  [ INFO ]         input name: data
  [ INFO ]         input type: f32
  [ INFO ]         input shape: {1, 3, 320, 544}
  [ INFO ]     outputs
  [ INFO ]         output name: detection_out
  [ INFO ]         output type: f32
  [ INFO ]         output shape: {1, 1, 200, 7}
  Reshape network to the image size = [960x1699]
  [ INFO ] model name: ResMobNet_v4 (LReLU) with single SSD head
  [ INFO ]     inputs
  [ INFO ]         input name: data
  [ INFO ]         input type: f32
  [ INFO ]         input shape: {1, 3, 960, 1699}
  [ INFO ]     outputs
  [ INFO ]         output name: detection_out
  [ INFO ]         output type: f32
  [ INFO ]         output shape: {1, 1, 200, 7}
  [0,1] element, prob = 0.716309,    (852,187)-(983,520)
  The resulting image was saved in the file: hello_reshape_ssd_output.bmp

  This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Integrate the OpenVINO™ Runtime with Your Application](../../../openvino-workflow/running-inference/index.md)
- [Get Started with Samples](get-started-demos.md)
- [Using OpenVINO Samples](index.md)
- [Convert a Model](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md)
- [Hello Reshape SSD Python Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_reshape_ssd/README.md)
- [Hello Reshape SSD C++ Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/hello_reshape_ssd/README.md)
