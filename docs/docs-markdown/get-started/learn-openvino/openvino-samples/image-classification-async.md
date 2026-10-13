---
sidebar_position: 6
sidebar_label: "Image Classification Async Sample"
description: "Learn how to do inference of image classification models using Asynchronous Inference Request API (Python, C++)."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Image Classification Async Sample

This sample demonstrates how to do inference of image classification models
using Asynchronous Inference Request API. Before using the sample, refer to the
following requirements:

- Models with only one input and output are supported.
- The sample accepts any file format supported by `core.read_model`.
- To build the sample, use instructions available at [Build the Sample Applications](get-started-demos.md#build-samples)
  section in "Get Started with Samples" guide.

## How It Works

At startup, the sample application reads command-line parameters, prepares input data, and
loads a specified model and an image to the OpenVINO™ Runtime plugin.
The batch size of the model is set according to the number of read images. The
batch mode is an independent attribute on the asynchronous mode.
The asynchronous mode works efficiently with any batch size.

Then, the sample creates an inference request object and assigns completion callback
for it. In scope of the completion callback handling, the inference request is executed again.

After that, the application starts inference for the first infer request and waits
until 10th inference request execution has been completed.
The asynchronous mode might increase the throughput of the pictures.

When inference is done, the application outputs data to the standard output stream.
You can place labels in `.labels` file near the model to get pretty output.

<Tabs>
  <TabItem value="python" label="Python">
  <div style={{maxHeight: '400px', overflowY: 'auto'}}>

  ```python
  #!/usr/bin/env python3
  # -*- coding: utf-8 -*-
  # Copyright (C) 2018-2026 Intel Corporation
  # SPDX-License-Identifier: Apache-2.0

  import argparse
  import logging as log
  import sys

  import cv2
  import numpy as np
  import openvino as ov


  def parse_args() -> argparse.Namespace:
      """Parse and return command line arguments."""
      parser = argparse.ArgumentParser(add_help=False)
      args = parser.add_argument_group('Options')
      # fmt: off
      args.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
      args.add_argument('-m', '--model', type=str, required=True,
                        help='Required. Path to an .xml or .onnx file with a trained model.')
      args.add_argument('-i', '--input', type=str, required=True, nargs='+',
                        help='Required. Path to an image file(s).')
      args.add_argument('-d', '--device', type=str, default='CPU',
                        help='Optional. Specify the target device to infer on; CPU, GPU or HETERO: '
                        'is acceptable. The sample will look for a suitable plugin for device specified. '
                        'Default value is CPU.')
      # fmt: on
      return parser.parse_args()


  def completion_callback(infer_request: ov.InferRequest, image_path: str) -> None:
      predictions = next(iter(infer_request.results.values()))

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


  def main() -> int:
      log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
      args = parse_args()

  # --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
      log.info('Creating OpenVINO Runtime Core')
      core = ov.Core()

  # --------------------------- Step 2. Read a model --------------------------------------------------------------------
      log.info(f'Reading the model: {args.model}')
      # (.xml and .bin files) or (.onnx file)
      model = core.read_model(args.model)

      if len(model.inputs) != 1:
          log.error('Sample supports only single input topologies')
          return -1

      if len(model.outputs) != 1:
          log.error('Sample supports only single output topologies')
          return -1

  # --------------------------- Step 3. Apply preprocessing -------------------------------------------------------------
      ppp = ov.preprocess.PrePostProcessor(model)

      # 1) Set input tensor information:
      # - input() provides information about a single model input
      # - precision of tensor is supposed to be 'u8'
      # - layout of data is 'NHWC'
      ppp.input().tensor() \
          .set_element_type(ov.Type.u8) \
          .set_layout(ov.Layout('NHWC'))  # noqa: N400

      # 2) Suppose model has 'NCHW' layout for input
      ppp.input().model().set_layout(ov.Layout('NCHW'))

      # 3) Set output tensor information:
      # - precision of tensor is supposed to be 'f32'
      ppp.output().tensor().set_element_type(ov.Type.f32)

      # 4) Apply preprocessing modifying the original 'model'
      model = ppp.build()

      # --------------------------- Step 4. Set up input --------------------------------------------------------------------
      # Read input images
      images = (cv2.imread(image_path) for image_path in args.input)

      # Resize images to model input dims
      _, h, w, _ = model.input().shape
      resized_images = (cv2.resize(image, (w, h)) for image in images)

      # Add N dimension
      input_tensors = (np.expand_dims(image, 0) for image in resized_images)

  # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
      log.info('Loading the model to the plugin')
      compiled_model = core.compile_model(model, args.device)

  # --------------------------- Step 6. Create infer request queue ------------------------------------------------------
      log.info('Starting inference in asynchronous mode')
      # create async queue with optimal number of infer requests
      infer_queue = ov.AsyncInferQueue(compiled_model)
      infer_queue.set_callback(completion_callback)

  # --------------------------- Step 7. Do inference --------------------------------------------------------------------
      for i, input_tensor in enumerate(input_tensors):
          infer_queue.start_async({0: input_tensor}, args.input[i])

      infer_queue.wait_all()
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

  /**
   * @brief The entry point the OpenVINO Runtime sample application
   * @file classification_sample_async/main.cpp
   * @example classification_sample_async/main.cpp
   */

  #include <sys/stat.h>

  #include <condition_variable>
  #include <exception>
  #include <fstream>
  #include <map>
  #include <memory>
  #include <mutex>
  #include <string>
  #include <vector>

  // clang-format off
  #include "openvino/openvino.hpp"

  #include "samples/args_helper.hpp"
  #include "samples/common.hpp"
  #include "samples/classification_results.h"
  #include "samples/slog.hpp"
  #include "format_reader_ptr.h"

  #include "classification_sample_async.h"
  // clang-format on

  using namespace ov::preprocess;

  namespace {
  bool parse_and_check_command_line(int argc, char* argv[]) {
      gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
      if (FLAGS_h) {
          show_usage();
          showAvailableDevices();
          return false;
      }
      slog::info << "Parsing input parameters" << slog::endl;

      if (FLAGS_m.empty()) {
          show_usage();
          throw std::logic_error("Model is required but not set. Please set -m option.");
      }

      if (FLAGS_i.empty()) {
          show_usage();
          throw std::logic_error("Input is required but not set. Please set -i option.");
      }

      return true;
  }
  }  // namespace

  int main(int argc, char* argv[]) {
      try {
          // -------- Get OpenVINO Runtime version --------
          slog::info << ov::get_openvino_version() << slog::endl;

          // -------- Parsing and validation of input arguments --------
          if (!parse_and_check_command_line(argc, argv)) {
              return EXIT_SUCCESS;
          }

          // -------- Read input --------
          // This vector stores paths to the processed images
          std::vector<std::string> image_names;
          parseInputFilesArguments(image_names);
          if (image_names.empty())
              throw std::logic_error("No suitable images were found");

          // -------- Step 1. Initialize OpenVINO Runtime Core --------
          ov::Core core;

          // -------- Step 2. Read a model --------
          slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;
          std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
          printInputAndOutputsInfo(*model);

          OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
          OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

          // -------- Step 3. Configure preprocessing --------
          const ov::Layout tensor_layout{"NHWC"};

          ov::preprocess::PrePostProcessor ppp(model);
          // 1) input() with no args assumes a model has a single input
          ov::preprocess::InputInfo& input_info = ppp.input();
          // 2) Set input tensor information:
          // - precision of tensor is supposed to be 'u8'
          // - layout of data is 'NHWC'
          input_info.tensor().set_element_type(ov::element::u8).set_layout(tensor_layout);
          // 3) Suppose model has 'NCHW' layout for input
          input_info.model().set_layout("NCHW");
          // 4) output() with no args assumes a model has a single result
          // - output() with no args assumes a model has a single result
          // - precision of tensor is supposed to be 'f32'
          ppp.output().tensor().set_element_type(ov::element::f32);

          // 5) Once the build() method is called, the pre(post)processing steps
          // for layout and precision conversions are inserted automatically
          model = ppp.build();

          // -------- Step 4. read input images --------
          slog::info << "Read input images" << slog::endl;

          ov::Shape input_shape = model->input().get_shape();
          const size_t width = input_shape[ov::layout::width_idx(tensor_layout)];
          const size_t height = input_shape[ov::layout::height_idx(tensor_layout)];

          std::vector<std::shared_ptr<unsigned char>> images_data;
          std::vector<std::string> valid_image_names;
          for (const auto& i : image_names) {
              FormatReader::ReaderPtr reader(i.c_str());
              if (reader.get() == nullptr) {
                  slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                  continue;
              }
              // Collect image data
              std::shared_ptr<unsigned char> data(reader->getData(width, height));
              if (data != nullptr) {
                  images_data.push_back(data);
                  valid_image_names.push_back(i);
              }
          }
          if (images_data.empty() || valid_image_names.empty())
              throw std::logic_error("Valid input images were not found!");

          // -------- Step 5. Setting batch size using image count --------
          const size_t batchSize = images_data.size();
          slog::info << "Set batch size " << std::to_string(batchSize) << slog::endl;
          ov::set_batch(model, batchSize);
          printInputAndOutputsInfo(*model);

          // -------- Step 6. Loading model to the device --------
          slog::info << "Loading model to the device " << FLAGS_d << slog::endl;
          ov::CompiledModel compiled_model = core.compile_model(model, FLAGS_d);

          // -------- Step 7. Create infer request --------
          slog::info << "Create infer request" << slog::endl;
          ov::InferRequest infer_request = compiled_model.create_infer_request();

          // -------- Step 8. Combine multiple input images as batch --------
          ov::Tensor input_tensor = infer_request.get_input_tensor();

          for (size_t image_id = 0; image_id < images_data.size(); ++image_id) {
              const size_t image_size = shape_size(model->input().get_shape()) / batchSize;
              std::memcpy(input_tensor.data<std::uint8_t>() + image_id * image_size,
                          images_data[image_id].get(),
                          image_size);
          }

          // -------- Step 9. Do asynchronous inference --------
          size_t num_iterations = 10;
          size_t cur_iteration = 0;
          std::condition_variable condVar;
          std::mutex mutex;
          std::exception_ptr exception_var;
          // -------- Step 10. Do asynchronous inference --------
          infer_request.set_callback([&](std::exception_ptr ex) {
              std::lock_guard<std::mutex> l(mutex);
              if (ex) {
                  exception_var = ex;
                  condVar.notify_all();
                  return;
              }

              cur_iteration++;
              slog::info << "Completed " << cur_iteration << " async request execution" << slog::endl;
              if (cur_iteration < num_iterations) {
                  // here a user can read output containing inference results and put new
                  // input to repeat async request again
                  infer_request.start_async();
              } else {
                  // continue sample execution after last Asynchronous inference request
                  // execution
                  condVar.notify_one();
              }
          });

          // Start async request for the first time
          slog::info << "Start inference (asynchronous executions)" << slog::endl;
          infer_request.start_async();

          // Wait all iterations of the async request
          std::unique_lock<std::mutex> lock(mutex);
          condVar.wait(lock, [&] {
              if (exception_var) {
                  std::rethrow_exception(exception_var);
              }

              return cur_iteration == num_iterations;
          });

          slog::info << "Completed async requests execution" << slog::endl;

          // -------- Step 11. Process output --------
          ov::Tensor output = infer_request.get_output_tensor();

          // Read labels from file (e.x. AlexNet.labels)
          std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
          std::vector<std::string> labels;

          std::ifstream inputFile;
          inputFile.open(labelFileName, std::ios::in);
          if (inputFile.is_open()) {
              std::string strLine;
              while (std::getline(inputFile, strLine)) {
                  trim(strLine);
                  labels.push_back(strLine);
              }
          }

          // Prints formatted classification results
          constexpr size_t N_TOP_RESULTS = 10;
          ClassificationResult classificationResult(output, valid_image_names, batchSize, N_TOP_RESULTS, labels);
          classificationResult.show();
      } catch (const std::exception& ex) {
          slog::err << ex.what() << slog::endl;
          return EXIT_FAILURE;
      } catch (...) {
          slog::err << "Unknown/internal exception happened." << slog::endl;
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

Run the application with the `-h` option to see the usage message:

<Tabs>
  <TabItem value="python" label="Python">
  ```console
  python classification_sample_async.py -h
  ```

  Usage message:

  ```console
  usage: classification_sample_async.py [-h] -m MODEL -i INPUT [INPUT ...]
                                        [-d DEVICE]

  Options:
    -h, --help            Show this help message and exit.
    -m MODEL, --model MODEL
                          Required. Path to an .xml or .onnx file with a trained
                          model.
    -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                          Required. Path to an image file(s).
    -d DEVICE, --device DEVICE
                          Optional. Specify the target device to infer on; CPU,
                          GPU or HETERO: is acceptable. The sample
                          will look for a suitable plugin for device specified.
                          Default value is CPU.
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```console
  classification_sample_async -h
  ```

  Usage instructions:

  ```console
  [ INFO ] OpenVINO Runtime version ......... <version>
  [ INFO ] Build ........... <build>

  classification_sample_async [OPTION]
  Options:

      -h                      Print usage instructions.
      -m "<path>"             Required. Path to an .xml file with a trained model.
      -i "<path>"             Required. Path to a folder with images or path to image files: a .ubyte file for LeNet and a .bmp file for other models.
      -d "<device>"           Optional. Specify the target device to infer on (the list of available devices is shown below). Default value is CPU. Use "-d HETERO:<comma_separated_devices_list>" format to specify the HETERO plugin. Sample will look for a suitable plugin for the device specified.

  Available target devices: <devices>
  ```
  </TabItem>
</Tabs>

To run the sample, you need to specify a model and an image:

- You can get a model specific for your inference task from one of model
  repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.
- You can use images from the media files collection available at
  [the storage](https://storage.openvinotoolkit.org/data/test_data).

:::note
- By default, OpenVINO™ Toolkit Samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using model conversion API with `reverse_input_channels` argument specified. For more information about the argument, refer to the **Color Conversion** section of [Preprocessing API](../../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details/index.md).
- Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using [model conversion API](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md).
- The sample accepts models in ONNX format (.onnx) that do not require preprocessing.
- The sample supports NCHW model layout only.
- When you specify single options multiple times, only the last value will be used. For example, the `-m` flag:
  
  `````{tab-set}
  ````{tab-item} Python
  :sync: python
  
  ```console
  python classification_sample_async.py -m model.xml -m model2.xml
  ```
  ````
  
  ````{tab-item} C++
  :sync: cpp
  
  ```console
  ./classification_sample_async -m model.xml -m model2.xml
  ```
  ````
  `````
:::

### Example

1. Download a pre-trained model:
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

4. Perform inference of image files, using a model on a `GPU`, for example:
   
   <Tabs>
     <TabItem value="python" label="Python">
     ```console
     python classification_sample_async.py -m ./models/alexnet.xml -i ./test_data/images/banana.jpg ./test_data/images/car.bmp -d GPU
     ```
     </TabItem>
     <TabItem value="cpp" label="C++">
     ```console
     classification_sample_async -m ./models/googlenet-v1.xml -i ./images/dog.bmp -d GPU
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
  [ INFO ] Reading the model: C:/test_data/models/alexnet.xml
  [ INFO ] Loading the model to the plugin
  [ INFO ] Starting inference in asynchronous mode
  [ INFO ] Image path: /test_data/images/banana.jpg
  [ INFO ] Top 10 results:
  [ INFO ] class_id probability
  [ INFO ] --------------------
  [ INFO ] 954      0.9707602
  [ INFO ] 666      0.0216788
  [ INFO ] 659      0.0032558
  [ INFO ] 435      0.0008082
  [ INFO ] 809      0.0004359
  [ INFO ] 502      0.0003860
  [ INFO ] 618      0.0002867
  [ INFO ] 910      0.0002866
  [ INFO ] 951      0.0002410
  [ INFO ] 961      0.0002193
  [ INFO ]
  [ INFO ] Image path: /test_data/images/car.bmp
  [ INFO ] Top 10 results:
  [ INFO ] class_id probability
  [ INFO ] --------------------
  [ INFO ] 656      0.5120340
  [ INFO ] 874      0.1142275
  [ INFO ] 654      0.0697167
  [ INFO ] 436      0.0615163
  [ INFO ] 581      0.0552262
  [ INFO ] 705      0.0304179
  [ INFO ] 675      0.0151660
  [ INFO ] 734      0.0151582
  [ INFO ] 627      0.0148493
  [ INFO ] 757      0.0120964
  [ INFO ]
  [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  The sample application logs each step in a standard output stream and
  outputs top-10 inference results.

  ```console
  [ INFO ] OpenVINO Runtime version ......... <version>
  [ INFO ] Build ........... <build>
  [ INFO ]
  [ INFO ] Parsing input parameters
  [ INFO ] Files were added: 1
  [ INFO ]     /images/dog.bmp
  [ INFO ] Loading model files:
  [ INFO ] /models/googlenet-v1.xml
  [ INFO ] model name: GoogleNet
  [ INFO ]     inputs
  [ INFO ]         input name: data
  [ INFO ]         input type: f32
  [ INFO ]         input shape: {1, 3, 224, 224}
  [ INFO ]     outputs
  [ INFO ]         output name: prob
  [ INFO ]         output type: f32
  [ INFO ]         output shape: {1, 1000}
  [ INFO ] Read input images
  [ INFO ] Set batch size 1
  [ INFO ] model name: GoogleNet
  [ INFO ]     inputs
  [ INFO ]         input name: data
  [ INFO ]         input type: u8
  [ INFO ]         input shape: {1, 224, 224, 3}
  [ INFO ]     outputs
  [ INFO ]         output name: prob
  [ INFO ]         output type: f32
  [ INFO ]         output shape: {1, 1000}
  [ INFO ] Loading model to the device GPU
  [ INFO ] Create infer request
  [ INFO ] Start inference (asynchronous executions)
  [ INFO ] Completed 1 async request execution
  [ INFO ] Completed 2 async request execution
  [ INFO ] Completed 3 async request execution
  [ INFO ] Completed 4 async request execution
  [ INFO ] Completed 5 async request execution
  [ INFO ] Completed 6 async request execution
  [ INFO ] Completed 7 async request execution
  [ INFO ] Completed 8 async request execution
  [ INFO ] Completed 9 async request execution
  [ INFO ] Completed 10 async request execution
  [ INFO ] Completed async requests execution

  Top 10 results:

  Image /images/dog.bmp

  classid probability
  ------- -----------
  156     0.8935547
  218     0.0608215
  215     0.0217133
  219     0.0105667
  212     0.0018835
  217     0.0018730
  152     0.0018730
  157     0.0015745
  154     0.0012817
  220     0.0010099
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Integrate the OpenVINO™ Runtime with Your Application](../../../openvino-workflow/running-inference/index.md)
- [Get Started with Samples](get-started-demos.md)
- [Using OpenVINO™ Toolkit Samples](index.md)
- [Convert a Model](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md)
- [Image Classification Async Python Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/python/classification_sample_async/README.md)
- [Image Classification Async C++ Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/classification_sample_async/README.md)
