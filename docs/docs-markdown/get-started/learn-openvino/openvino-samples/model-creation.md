---
sidebar_position: 7
sidebar_label: "Model Creation Sample"
description: "Learn how to create a model on the fly with a provided weights file and infer it later using Synchronous Inference Request API (Python, C++)."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Model Creation Sample

This sample demonstrates how to run inference using a [model](../../../openvino-workflow/running-inference/model-representation.md)
built on the fly that uses weights from the LeNet classification model, which is
known to work well on digit classification tasks. You do not need an XML file,
the model is created from the source code on the fly. Before using the sample,
refer to the following requirements:

- The sample accepts a model weights file (\*.bin).
- The sample has been validated with a LeNet model.
- To build the sample, use instructions available at [Build the Sample Applications](get-started-demos.md#build-samples)
  section in "Get Started with Samples" guide.

## How It Works

At startup, the sample application reads command-line parameters, [builds a model](../../../openvino-workflow/running-inference/model-representation.md)
and passes the weights file. Then, it loads the model and input data to the OpenVINO™
Runtime plugin. Finally, it performs synchronous inference and processes output
data, logging each step in a standard output stream.

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
  from functools import reduce

  import numpy as np
  import openvino as ov
  from openvino import op, opset1, opset8

  from data import digits


  def create_model(model_path: str) -> ov.Model:
      """Create a model on the fly from the source code using openvino."""

      def shape_and_length(shape: list) -> tuple[list, int]:
          length = reduce(lambda x, y: x * y, shape)
          return shape, length

      weights = np.fromfile(model_path, dtype=np.float32)
      weights_offset = 0
      padding_begin = padding_end = [0, 0]

      # input
      input_shape = [64, 1, 28, 28]
      param_node = op.Parameter(ov.Type.f32, ov.Shape(input_shape))

      # convolution 1
      conv_1_kernel_shape, conv_1_kernel_length = shape_and_length([20, 1, 5, 5])
      conv_1_kernel = op.Constant(ov.Type.f32, ov.Shape(conv_1_kernel_shape), weights[0:conv_1_kernel_length].tolist())
      weights_offset += conv_1_kernel_length
      conv_1_node = opset8.convolution(param_node, conv_1_kernel, [1, 1], padding_begin, padding_end, [1, 1])

      # add 1
      add_1_kernel_shape, add_1_kernel_length = shape_and_length([1, 20, 1, 1])
      add_1_kernel = op.Constant(ov.Type.f32, ov.Shape(add_1_kernel_shape),
                                 weights[weights_offset : weights_offset + add_1_kernel_length])
      weights_offset += add_1_kernel_length
      add_1_node = opset8.add(conv_1_node, add_1_kernel)

      # maxpool 1
      maxpool_1_node = opset1.max_pool(add_1_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil')

      # convolution 2
      conv_2_kernel_shape, conv_2_kernel_length = shape_and_length([50, 20, 5, 5])
      conv_2_kernel = op.Constant(ov.Type.f32, ov.Shape(conv_2_kernel_shape),
                                  weights[weights_offset : weights_offset + conv_2_kernel_length],
                                  )
      weights_offset += conv_2_kernel_length
      conv_2_node = opset8.convolution(maxpool_1_node, conv_2_kernel, [1, 1], padding_begin, padding_end, [1, 1])

      # add 2
      add_2_kernel_shape, add_2_kernel_length = shape_and_length([1, 50, 1, 1])
      add_2_kernel = op.Constant(ov.Type.f32, ov.Shape(add_2_kernel_shape),
                                 weights[weights_offset : weights_offset + add_2_kernel_length],
                                 )
      weights_offset += add_2_kernel_length
      add_2_node = opset8.add(conv_2_node, add_2_kernel)

      # maxpool 2
      maxpool_2_node = opset1.max_pool(add_2_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil')

      # reshape 1
      reshape_1_dims, reshape_1_length = shape_and_length([2])
      # workaround to get int64 weights from float32 ndarray w/o unnecessary copying
      dtype_weights = np.frombuffer(
          weights[weights_offset : weights_offset + 2 * reshape_1_length],
          dtype=np.int64,
      )
      reshape_1_kernel = op.Constant(ov.Type.i64, ov.Shape(list(dtype_weights.shape)), dtype_weights)
      weights_offset += 2 * reshape_1_length
      reshape_1_node = opset8.reshape(maxpool_2_node, reshape_1_kernel, True)

      # matmul 1
      matmul_1_kernel_shape, matmul_1_kernel_length = shape_and_length([500, 800])
      matmul_1_kernel = op.Constant(ov.Type.f32, ov.Shape(matmul_1_kernel_shape),
                                    weights[weights_offset : weights_offset + matmul_1_kernel_length],
                                    )
      weights_offset += matmul_1_kernel_length
      matmul_1_node = opset8.matmul(reshape_1_node, matmul_1_kernel, False, True)

      # add 3
      add_3_kernel_shape, add_3_kernel_length = shape_and_length([1, 500])
      add_3_kernel = op.Constant(ov.Type.f32, ov.Shape(add_3_kernel_shape),
                                 weights[weights_offset : weights_offset + add_3_kernel_length],
                                 )
      weights_offset += add_3_kernel_length
      add_3_node = opset8.add(matmul_1_node, add_3_kernel)

      # ReLU
      relu_node = opset8.relu(add_3_node)

      # reshape 2
      reshape_2_kernel = op.Constant(ov.Type.i64, ov.Shape(list(dtype_weights.shape)), dtype_weights)
      reshape_2_node = opset8.reshape(relu_node, reshape_2_kernel, True)

      # matmul 2
      matmul_2_kernel_shape, matmul_2_kernel_length = shape_and_length([10, 500])
      matmul_2_kernel = op.Constant(ov.Type.f32, ov.Shape(matmul_2_kernel_shape),
                                    weights[weights_offset : weights_offset + matmul_2_kernel_length],
                                    )
      weights_offset += matmul_2_kernel_length
      matmul_2_node = opset8.matmul(reshape_2_node, matmul_2_kernel, False, True)

      # add 4
      add_4_kernel_shape, add_4_kernel_length = shape_and_length([1, 10])
      add_4_kernel = op.Constant(ov.Type.f32, ov.Shape(add_4_kernel_shape),
                                 weights[weights_offset : weights_offset + add_4_kernel_length],
                                 )
      weights_offset += add_4_kernel_length
      add_4_node = opset8.add(matmul_2_node, add_4_kernel)

      # softmax
      softmax_axis = 1
      softmax_node = opset8.softmax(add_4_node, softmax_axis)

      return ov.Model(softmax_node, [param_node], 'lenet')


  def main():
      log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
      # Parsing and validation of input arguments
      if len(sys.argv) != 3:
          log.info(f'Usage: {sys.argv[0]} <path_to_model> <device_name>')
          return 1

      model_path = sys.argv[1]
      device_name = sys.argv[2]
      labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
      number_top = 1
      # ---------------------------Step 1. Initialize OpenVINO Runtime Core--------------------------------------------------
      log.info('Creating OpenVINO Runtime Core')

      # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation------------------------------
      log.info(f'Loading the model using openvino with weights from {model_path}')
      model = create_model(model_path)
      # ---------------------------Step 3. Apply preprocessing----------------------------------------------------------
      # Get names of input and output blobs
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

      # Set a batch size equal to number of input images
      ov.set_batch(model, digits.shape[0])
      # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
      log.info('Loading the model to the plugin')
      core = ov.Core()
      compiled_model = core.compile_model(model, device_name)

      # ---------------------------Step 5. Prepare input---------------------------------------------------------------------
      n, c, h, w = model.input().shape
      input_data = np.ndarray(shape=(n, c, h, w))
      for i in range(n):
          image = digits[i].reshape(28, 28)
          image = image[:, :, np.newaxis]
          input_data[i] = image

      # ---------------------------Step 6. Do inference----------------------------------------------------------------------
      log.info('Starting inference in synchronous mode')
      results = compiled_model.infer_new_request({0: input_data})

      # ---------------------------Step 7. Process output--------------------------------------------------------------------
      predictions = next(iter(results.values()))

      log.info(f'Top {number_top} results: ')
      for i in range(n):
          probs = predictions[i]
          # Get an array of number_top class IDs in descending order of probability
          top_n_idexes = np.argsort(probs)[-number_top :][::-1]

          header = 'classid probability'
          header = header + ' label' if labels else header

          log.info(f'Image {i}')
          log.info('')
          log.info(header)
          log.info('-' * len(header))

          for class_id in top_n_idexes:
              probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
              label_indent = ' ' * (len('probability') - 8) if labels else ''
              label = labels[class_id] if labels else ''
              log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}{label_indent}{label}')
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

  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <limits>
  #include <memory>
  #include <string>
  #include <vector>

  // clang-format off
  #include "openvino/openvino.hpp"
  #include "openvino/opsets/opset13.hpp"

  #include "samples/args_helper.hpp"
  #include "samples/common.hpp"
  #include "samples/classification_results.h"
  #include "samples/slog.hpp"

  #include "model_creation_sample.hpp"
  // clang-format on

  constexpr auto N_TOP_RESULTS = 1;
  constexpr auto LENET_WEIGHTS_SIZE = 1724336;
  constexpr auto LENET_NUM_CLASSES = 10;

  using namespace ov;
  using namespace ov::preprocess;

  /**
   * @brief Read file to the buffer
   * @param file_name string
   * @param buffer to store file content
   * @param maxSize length of file
   * @return none
   */
  void read_file(const std::string& file_name, void* buffer, size_t maxSize) {
      std::ifstream input_file;

      input_file.open(file_name, std::ios::binary | std::ios::in);
      if (!input_file.is_open()) {
          throw std::logic_error("Cannot open weights file");
      }

      if (!input_file.read(reinterpret_cast<char*>(buffer), maxSize)) {
          input_file.close();
          throw std::logic_error("Cannot read bytes from weights file");
      }

      input_file.close();
  }

  /**
   * @brief Read .bin file with weights for the trained model
   * @param filepath string
   * @return weightsPtr tensor blob
   */
  ov::Tensor read_weights(const std::string& filepath) {
      std::ifstream weightFile(filepath, std::ifstream::ate | std::ifstream::binary);

      int64_t fileSize = weightFile.tellg();
      OPENVINO_ASSERT(fileSize == LENET_WEIGHTS_SIZE,
                      "Incorrect weights file. This sample works only with LeNet "
                      "classification model.");

      ov::Tensor weights(ov::element::u8, {static_cast<size_t>(fileSize)});
      read_file(filepath, weights.data(), weights.get_byte_size());

      return weights;
  }

  /**
   * @brief Create ov::Model
   * @return Ptr to ov::Model
   */
  std::shared_ptr<ov::Model> create_model(const std::string& path_to_weights) {
      const ov::Tensor weights = read_weights(path_to_weights);
      const std::uint8_t* data = weights.data<std::uint8_t>();

      // -------input------
      std::vector<ptrdiff_t> padBegin{0, 0};
      std::vector<ptrdiff_t> padEnd{0, 0};

      auto paramNode = std::make_shared<ov::opset13::Parameter>(ov::element::Type_t::f32, ov::Shape({64, 1, 28, 28}));

      // -------convolution 1----
      auto convFirstShape = Shape{20, 1, 5, 5};
      auto convolutionFirstConstantNode = std::make_shared<opset13::Constant>(element::Type_t::f32, convFirstShape, data);

      auto convolutionNodeFirst = std::make_shared<opset13::Convolution>(paramNode->output(0),
                                                                         convolutionFirstConstantNode->output(0),
                                                                         Strides({1, 1}),
                                                                         CoordinateDiff(padBegin),
                                                                         CoordinateDiff(padEnd),
                                                                         Strides({1, 1}));

      // -------Add--------------
      auto addFirstShape = Shape{1, 20, 1, 1};
      auto offset = shape_size(convFirstShape) * sizeof(float);
      auto addFirstConstantNode = std::make_shared<opset13::Constant>(element::Type_t::f32, addFirstShape, data + offset);

      auto addNodeFirst =
          std::make_shared<opset13::Add>(convolutionNodeFirst->output(0), addFirstConstantNode->output(0));

      // -------MAXPOOL----------
      Shape padBeginShape{0, 0};
      Shape padEndShape{0, 0};

      auto maxPoolingNodeFirst = std::make_shared<opset13::MaxPool>(addNodeFirst->output(0),
                                                                    Strides{2, 2},
                                                                    Strides{1, 1},
                                                                    padBeginShape,
                                                                    padEndShape,
                                                                    Shape{2, 2},
                                                                    op::RoundingType::CEIL);

      // -------convolution 2----
      auto convSecondShape = Shape{50, 20, 5, 5};
      offset += shape_size(addFirstShape) * sizeof(float);
      auto convolutionSecondConstantNode =
          std::make_shared<opset13::Constant>(element::Type_t::f32, convSecondShape, data + offset);

      auto convolutionNodeSecond = std::make_shared<opset13::Convolution>(maxPoolingNodeFirst->output(0),
                                                                          convolutionSecondConstantNode->output(0),
                                                                          Strides({1, 1}),
                                                                          CoordinateDiff(padBegin),
                                                                          CoordinateDiff(padEnd),
                                                                          Strides({1, 1}));

      // -------Add 2------------
      auto addSecondShape = Shape{1, 50, 1, 1};
      offset += shape_size(convSecondShape) * sizeof(float);
      auto addSecondConstantNode =
          std::make_shared<opset13::Constant>(element::Type_t::f32, addSecondShape, data + offset);

      auto addNodeSecond =
          std::make_shared<opset13::Add>(convolutionNodeSecond->output(0), addSecondConstantNode->output(0));

      // -------MAXPOOL 2--------
      auto maxPoolingNodeSecond = std::make_shared<opset13::MaxPool>(addNodeSecond->output(0),
                                                                     Strides{2, 2},
                                                                     Strides{1, 1},
                                                                     padBeginShape,
                                                                     padEndShape,
                                                                     Shape{2, 2},
                                                                     op::RoundingType::CEIL);

      // -------Reshape----------
      auto reshapeFirstShape = Shape{2};
      auto reshapeOffset = shape_size(addSecondShape) * sizeof(float) + offset;
      auto reshapeFirstConstantNode =
          std::make_shared<opset13::Constant>(element::Type_t::i64, reshapeFirstShape, data + reshapeOffset);

      auto reshapeFirstNode =
          std::make_shared<opset13::Reshape>(maxPoolingNodeSecond->output(0), reshapeFirstConstantNode->output(0), true);

      // -------MatMul 1---------
      auto matMulFirstShape = Shape{500, 800};
      offset = shape_size(reshapeFirstShape) * sizeof(int64_t) + reshapeOffset;
      auto matMulFirstConstantNode =
          std::make_shared<opset13::Constant>(element::Type_t::f32, matMulFirstShape, data + offset);

      auto matMulFirstNode =
          std::make_shared<opset13::MatMul>(reshapeFirstNode->output(0), matMulFirstConstantNode->output(0), false, true);

      // -------Add 3------------
      auto addThirdShape = Shape{1, 500};
      offset += shape_size(matMulFirstShape) * sizeof(float);
      auto addThirdConstantNode = std::make_shared<opset13::Constant>(element::Type_t::f32, addThirdShape, data + offset);

      auto addThirdNode = std::make_shared<opset13::Add>(matMulFirstNode->output(0), addThirdConstantNode->output(0));

      // -------Relu-------------
      auto reluNode = std::make_shared<opset13::Relu>(addThirdNode->output(0));

      // -------Reshape 2--------
      auto reshapeSecondShape = Shape{2};
      auto reshapeSecondConstantNode =
          std::make_shared<opset13::Constant>(element::Type_t::i64, reshapeSecondShape, data + reshapeOffset);

      auto reshapeSecondNode =
          std::make_shared<opset13::Reshape>(reluNode->output(0), reshapeSecondConstantNode->output(0), true);

      // -------MatMul 2---------
      auto matMulSecondShape = Shape{10, 500};
      offset += shape_size(addThirdShape) * sizeof(float);
      auto matMulSecondConstantNode =
          std::make_shared<opset13::Constant>(element::Type_t::f32, matMulSecondShape, data + offset);

      auto matMulSecondNode = std::make_shared<opset13::MatMul>(reshapeSecondNode->output(0),
                                                                matMulSecondConstantNode->output(0),
                                                                false,
                                                                true);

      // -------Add 4------------
      auto add4Shape = Shape{1, 10};
      offset += shape_size(matMulSecondShape) * sizeof(float);
      auto add4ConstantNode = std::make_shared<opset13::Constant>(element::Type_t::f32, add4Shape, data + offset);

      auto add4Node = std::make_shared<opset13::Add>(matMulSecondNode->output(0), add4ConstantNode->output(0));

      // -------softMax----------
      auto softMaxNode = std::make_shared<opset13::Softmax>(add4Node->output(0), 1);
      softMaxNode->get_output_tensor(0).set_names({"output_tensor"});

      // ------- OpenVINO function--
      auto result_full = std::make_shared<opset13::Result>(softMaxNode->output(0));

      std::shared_ptr<ov::Model> fnPtr =
          std::make_shared<ov::Model>(result_full, ov::ParameterVector{paramNode}, "lenet");

      return fnPtr;
  }

  /**
   * @brief The entry point for OpenVINO ov::Model creation sample
   */
  int main(int argc, char* argv[]) {
      try {
          // -------- Get OpenVINO runtime version --------
          slog::info << ov::get_openvino_version() << slog::endl;

          // -------- Parsing and validation of input arguments --------
          if (argc != 3) {
              std::cout << "Usage : " << argv[0] << " <path_to_lenet_weights> <device>" << std::endl;
              return EXIT_FAILURE;
          }
          const std::string weights_path{argv[1]};
          const std::string device_name{argv[2]};

          // -------- Step 1. Initialize OpenVINO Runtime Core object --------
          ov::Core core;

          slog::info << "Device info: " << slog::endl;
          slog::info << core.get_versions(device_name) << slog::endl;

          // -------- Step 2. Create network using ov::Function --------
          slog::info << "Create model from weights: " << weights_path << slog::endl;
          std::shared_ptr<ov::Model> model = create_model(weights_path);
          printInputAndOutputsInfo(*model);

          OPENVINO_ASSERT(model->inputs().size() == 1, "Incorrect number of inputs for LeNet");
          OPENVINO_ASSERT(model->outputs().size() == 1, "Incorrect number of outputs for LeNet");

          ov::Shape input_shape = model->input().get_shape();
          OPENVINO_ASSERT(input_shape.size() == 4, "Incorrect input dimensions for LeNet");

          const ov::Shape output_shape = model->output().get_shape();
          OPENVINO_ASSERT(output_shape.size() == 2, "Incorrect output dimensions for LeNet");

          const auto classCount = output_shape[1];
          OPENVINO_ASSERT(classCount <= LENET_NUM_CLASSES, "Incorrect number of output classes for LeNet model");

          // -------- Step 3. Apply preprocessing --------
          const Layout tensor_layout{"NHWC"};

          // apply preprocessing
          ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

          // 1) InputInfo() with no args assumes a model has a single input
          ov::preprocess::InputInfo& input_info = ppp.input();
          // 2) Set input tensor information:
          // - layout of data is 'NHWC'
          // - precision of tensor is supposed to be 'u8'
          input_info.tensor().set_layout(tensor_layout).set_element_type(element::u8);
          // 3) Here we suppose model has 'NCHW' layout for input
          input_info.model().set_layout("NCHW");

          // 4) Once the build() method is called, the preprocessing steps
          // for layout and precision conversions are inserted automatically
          model = ppp.build();

          // Set batch size using images count
          const size_t batch_size = digits.size();

          // -------- Step 4. Reshape a model to new batch size --------
          // Setting batch size using image count
          ov::set_batch(model, batch_size);
          slog::info << "Batch size is " << std::to_string(batch_size) << slog::endl;
          printInputAndOutputsInfo(*model);

          // -------- Step 5. Compiling model for the device --------
          slog::info << "Compiling a model for the " << device_name << " device" << slog::endl;
          ov::CompiledModel compiled_model = core.compile_model(model, device_name);

          // -------- Step 6. Create infer request --------
          slog::info << "Create infer request" << slog::endl;
          ov::InferRequest infer_request = compiled_model.create_infer_request();

          // -------- Step 7. Combine multiple input images as batch --------
          slog::info << "Combine images in batch and set to input tensor" << slog::endl;
          ov::Tensor input_tensor = infer_request.get_input_tensor();

          // Iterate over all input images and copy data to input tensor
          for (size_t image_id = 0; image_id < digits.size(); ++image_id) {
              const size_t image_size = shape_size(model->input().get_shape()) / batch_size;
              std::memcpy(input_tensor.data<std::uint8_t>() + image_id * image_size, digits[image_id], image_size);
          }

          // -------- Step 8. Do sync inference --------
          slog::info << "Start sync inference" << slog::endl;
          infer_request.infer();

          // -------- Step 9. Process output --------
          slog::info << "Processing output tensor" << slog::endl;
          const ov::Tensor output_tensor = infer_request.get_output_tensor();

          const std::vector<std::string> lenet_labels{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

          // Prints formatted classification results
          ClassificationResult classification_result(output_tensor,
                                                     lenet_labels,  // in this sample images have the same names as labels
                                                     batch_size,
                                                     N_TOP_RESULTS,
                                                     lenet_labels);
          classification_result.show();
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

You can see the explicit description of each sample step at [Integration Steps](../../../openvino-workflow/running-inference/index.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Running

To run the sample, you need to specify model weights and a device.

<Tabs>
  <TabItem value="python" label="Python">
  ```console
  python model_creation_sample.py <path_to_weights_file> <device_name>
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  model_creation_sample <path_to_weights_file> <device_name>
  ```
  </TabItem>
</Tabs>

:::note
- This sample supports models with FP32 weights only.
- The `lenet.bin` weights file is generated by
  [model conversion API](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md)
  from the public LeNet model, with the `input_shape [64,1,28,28]` parameter specified.
- The original model is available in the
  [Caffe repository](https://github.com/BVLC/caffe/tree/master/examples/mnist) on GitHub.
:::

### Example

<Tabs>
  <TabItem value="python" label="Python">
  ```console
  python model_creation_sample.py lenet.bin GPU
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  model_creation_sample lenet.bin GPU
  ```
  </TabItem>
</Tabs>

## Sample Output

<Tabs>
  <TabItem value="python" label="Python">
  The sample application logs each step in a standard output stream and outputs 10 inference results.

  ```console
  [ INFO ] Creating OpenVINO Runtime Core
  [ INFO ] Loading the model using openvino with weights from lenet.bin
  [ INFO ] Loading the model to the plugin
  [ INFO ] Starting inference in synchronous mode
  [ INFO ] Top 1 results:
  [ INFO ] Image 0
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 0       1.0000000   0
  [ INFO ]
  [ INFO ] Image 1
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 1       1.0000000   1
  [ INFO ]
  [ INFO ] Image 2
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 2       1.0000000   2
  [ INFO ]
  [ INFO ] Image 3
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 3       1.0000000   3
  [ INFO ]
  [ INFO ] Image 4
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 4       1.0000000   4
  [ INFO ]
  [ INFO ] Image 5
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 5       1.0000000   5
  [ INFO ]
  [ INFO ] Image 6
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 6       1.0000000   6
  [ INFO ]
  [ INFO ] Image 7
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 7       1.0000000   7
  [ INFO ]
  [ INFO ] Image 8
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 8       1.0000000   8
  [ INFO ]
  [ INFO ] Image 9
  [ INFO ]
  [ INFO ] classid probability label
  [ INFO ] -------------------------
  [ INFO ] 9       1.0000000   9
  [ INFO ]
  [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  The sample application logs each step in a standard output stream and outputs top-10 inference results.

  ```console
  [ INFO ] OpenVINO Runtime version ......... <version>
  [ INFO ] Build ........... <build>
  [ INFO ]
  [ INFO ] Device info:
  [ INFO ] GPU
  [ INFO ] Intel GPU plugin version ......... <version>
  [ INFO ] Build ........... <build>
  [ INFO ]
  [ INFO ]
  [ INFO ] Create model from weights: lenet.bin
  [ INFO ] model name: lenet
  [ INFO ]     inputs
  [ INFO ]         input name: NONE
  [ INFO ]         input type: f32
  [ INFO ]         input shape: {64, 1, 28, 28}
  [ INFO ]     outputs
  [ INFO ]         output name: output_tensor
  [ INFO ]         output type: f32
  [ INFO ]         output shape: {64, 10}
  [ INFO ] Batch size is 10
  [ INFO ] model name: lenet
  [ INFO ]     inputs
  [ INFO ]         input name: NONE
  [ INFO ]         input type: u8
  [ INFO ]         input shape: {10, 28, 28, 1}
  [ INFO ]     outputs
  [ INFO ]         output name: output_tensor
  [ INFO ]         output type: f32
  [ INFO ]         output shape: {10, 10}
  [ INFO ] Compiling a model for the GPU device
  [ INFO ] Create infer request
  [ INFO ] Combine images in batch and set to input tensor
  [ INFO ] Start sync inference
  [ INFO ] Processing output tensor

  Top 1 results:

  Image 0

  classid probability label
  ------- ----------- -----
  0       1.0000000   0

  Image 1

  classid probability label
  ------- ----------- -----
  1       1.0000000   1

  Image 2

  classid probability label
  ------- ----------- -----
  2       1.0000000   2

  Image 3

  classid probability label
  ------- ----------- -----
  3       1.0000000   3

  Image 4

  classid probability label
  ------- ----------- -----
  4       1.0000000   4

  Image 5

  classid probability label
  ------- ----------- -----
  5       1.0000000   5

  Image 6

  classid probability label
  ------- ----------- -----
  6       1.0000000   6

  Image 7

  classid probability label
  ------- ----------- -----
  7       1.0000000   7

  Image 8

  classid probability label
  ------- ----------- -----
  8       1.0000000   8

  Image 9

  classid probability label
  ------- ----------- -----
  9       1.0000000   9
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Integrate the OpenVINO™ Runtime with Your Application](../../../openvino-workflow/running-inference/index.md)
- [Get Started with Samples](get-started-demos.md)
- [Using OpenVINO Samples](index.md)
- [Convert a Model](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md)
- [Model Creation Python Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/model_creation_sample/README.md)
- [Model Creation C++ Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/model_creation_sample/README.md)
