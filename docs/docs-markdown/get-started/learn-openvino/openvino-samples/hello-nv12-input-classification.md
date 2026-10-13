---
sidebar_position: 3
sidebar_label: "Hello NV12 Input Classification Sample"
description: "Learn how to do inference of image classification models with images in NV12 color format using Synchronous Inference Request (C++) API."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hello NV12 Input Classification Sample

This sample demonstrates how to execute an inference of image classification models
with images in NV12 color format using Synchronous Inference Request API. Before
using the sample, refer to the following requirements:

- The sample accepts any file format supported by `ov::Core::read_model`.
- To build the sample, use instructions available at [Build the Sample Applications](get-started-demos.md#build-samples)
  section in "Get Started with Samples" guide.

## How It Works

At startup, the sample application reads command line parameters, loads the
specified model and an image in the NV12 color format to an OpenVINO™ Runtime
plugin. Then, the sample creates an synchronous inference request object. When
inference is done, the application outputs data to the standard output stream.
You can place labels in `.labels` file near the model to get pretty output.

<Tabs>
  <TabItem value="cpp" label="C++">
  <div style={{maxHeight: '400px', overflowY: 'auto'}}>

  ```cpp
  // Copyright (C) 2018-2026 Intel Corporation
  // SPDX-License-Identifier: Apache-2.0
  //

  #include <sys/stat.h>

  #include <cassert>
  #include <fstream>
  #include <iostream>
  #include <memory>
  #include <sstream>
  #include <string>
  #include <utility>
  #include <vector>
  #ifdef _WIN32
  #    include "samples/os/windows/w_dirent.h"
  #else
  #    include <dirent.h>
  #endif

  // clang-format off
  #include "openvino/openvino.hpp"

  #include "samples/args_helper.hpp"
  #include "samples/common.hpp"
  #include "samples/slog.hpp"
  #include "samples/classification_results.h"
  #include "format_reader_ptr.h"
  // clang-format on

  constexpr auto N_TOP_RESULTS = 10;

  using namespace ov::preprocess;

  /**
   * @brief Parse image size provided as string in format WIDTHxHEIGHT
   * @param string of image size in WIDTHxHEIGHT format
   * @return parsed width and height
   */
  std::pair<size_t, size_t> parse_image_size(const std::string& size_string) {
      auto delimiter_pos = size_string.find("x");
      if (delimiter_pos == std::string::npos || delimiter_pos >= size_string.size() - 1 || delimiter_pos == 0) {
          std::stringstream err;
          err << "Incorrect format of image size parameter, expected WIDTHxHEIGHT, "
                 "actual: "
              << size_string;
          throw std::runtime_error(err.str());
      }

      size_t width = static_cast<size_t>(std::stoull(size_string.substr(0, delimiter_pos)));
      size_t height = static_cast<size_t>(std::stoull(size_string.substr(delimiter_pos + 1, size_string.size())));

      if (width == 0 || height == 0) {
          throw std::runtime_error("Incorrect format of image size parameter, width "
                                   "and height must not be equal to 0");
      }

      if (width % 2 != 0 || height % 2 != 0) {
          throw std::runtime_error("Unsupported image size, width and height must be even numbers");
      }

      return {width, height};
  }

  /**
   * @brief The entry point of the OpenVINO Runtime sample application
   */
  int main(int argc, char* argv[]) {
      try {
          // -------- Get OpenVINO runtime version --------
          slog::info << ov::get_openvino_version() << slog::endl;

          // -------- Parsing and validation input arguments --------
          if (argc != 5) {
              std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <image_size> <device_name>"
                        << std::endl;
              return EXIT_FAILURE;
          }

          const std::string model_path{argv[1]};
          const std::string image_path{argv[2]};
          size_t input_width = 0;
          size_t input_height = 0;
          std::tie(input_width, input_height) = parse_image_size(argv[3]);
          const std::string device_name{argv[4]};
          // -----------------------------------------------------------------------------------------------------

          // -------- Read image names --------
          FormatReader::ReaderPtr reader(image_path.c_str());
          if (reader.get() == nullptr) {
              std::string msg = "Image " + image_path + " cannot be read!";
              throw std::logic_error(msg);
          }

          size_t batch = 1;

          // -----------------------------------------------------------------------------------------------------

          // -------- Step 1. Initialize OpenVINO Runtime Core ---------
          ov::Core core;

          // -------- Step 2. Read a model --------
          slog::info << "Loading model files: " << model_path << slog::endl;
          std::shared_ptr<ov::Model> model = core.read_model(model_path);
          printInputAndOutputsInfo(*model);

          OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
          OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

          std::string input_tensor_name = model->input().get_any_name();
          std::string output_tensor_name = model->output().get_any_name();

          // -------- Step 3. Configure preprocessing  --------
          PrePostProcessor ppp = PrePostProcessor(model);

          // 1) Select input with 'input_tensor_name' tensor name
          InputInfo& input_info = ppp.input(input_tensor_name);
          // 2) Set input type
          // - as 'u8' precision
          // - set color format to NV12 (single plane)
          // - static spatial dimensions for resize preprocessing operation
          input_info.tensor()
              .set_element_type(ov::element::u8)
              .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
              .set_spatial_static_shape(input_height, input_width);
          // 3) Pre-processing steps:
          //    a) Convert to 'float'. This is to have color conversion more accurate
          //    b) Convert to BGR: Assumes that model accepts images in BGR format. For RGB, change it manually
          //    c) Resize image from tensor's dimensions to model ones
          input_info.preprocess()
              .convert_element_type(ov::element::f32)
              .convert_color(ColorFormat::BGR)
              .resize(ResizeAlgorithm::RESIZE_LINEAR);
          // 4) Set model data layout (Assuming model accepts images in NCHW layout)
          input_info.model().set_layout("NCHW");

          // 5) Apply preprocessing to an input with 'input_tensor_name' name of loaded model
          model = ppp.build();

          // -------- Step 4. Loading a model to the device --------
          ov::CompiledModel compiled_model = core.compile_model(model, device_name);

          // -------- Step 5. Create an infer request --------
          ov::InferRequest infer_request = compiled_model.create_infer_request();

          // -------- Step 6. Prepare input data  --------
          std::shared_ptr<unsigned char> image_data = reader->getData(input_width, input_height);

          ov::Tensor input_tensor{ov::element::u8, {batch, input_height * 3 / 2, input_width, 1}, image_data.get()};

          // Read labels from file (e.x. AlexNet.labels)
          std::string labelFileName = fileNameNoExt(model_path) + ".labels";
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

          // -------- Step 7. Set input tensor  --------
          // Set the input tensor by tensor name to the InferRequest
          infer_request.set_tensor(input_tensor_name, input_tensor);

          // -------- Step 8. Do inference --------
          // Running the request synchronously
          infer_request.infer();

          // -------- Step 9. Process output --------
          ov::Tensor output = infer_request.get_tensor(output_tensor_name);

          // Print classification results
          ClassificationResult classification_result(output, {image_path}, batch, N_TOP_RESULTS, labels);
          classification_result.show();

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
  #include "openvino/c/openvino.h"

  /**
   * @brief Struct to store infer results
   */
  struct infer_result {
      size_t class_id;
      float probability;
  };

  /**
   * @brief Sort result by probability
   * @param struct with infer results to sort
   * @param result_size of the struct
   * @return none
   */
  int compare(const void* a, const void* b) {
      const struct infer_result* sa = (const struct infer_result*)a;
      const struct infer_result* sb = (const struct infer_result*)b;
      if (sa->probability < sb->probability) {
          return 1;
      } else if ((sa->probability == sb->probability) && (sa->class_id > sb->class_id)) {
          return 1;
      } else if (sa->probability > sb->probability) {
          return -1;
      }
      return 0;
  }
  void infer_result_sort(struct infer_result* results, size_t result_size) {
      qsort(results, result_size, sizeof(struct infer_result), compare);
  }

  /**
   * @brief Convert output tensor to infer result struct for processing results
   * @param tensor of output tensor
   * @param result_size of the infer result
   * @return struct infer_result
   */
  struct infer_result* tensor_to_infer_result(ov_tensor_t* tensor, size_t* result_size) {
      ov_status_e status = ov_tensor_get_size(tensor, result_size);
      if (status != OK)
          return NULL;

      struct infer_result* results = (struct infer_result*)malloc(sizeof(struct infer_result) * (*result_size));
      if (!results)
          return NULL;

      void* data = NULL;
      status = ov_tensor_data(tensor, &data);
      if (status != OK) {
          free(results);
          return NULL;
      }

      float* float_data = (float*)(data);
      for (size_t i = 0; i < *result_size; ++i) {
          results[i].class_id = i;
          results[i].probability = float_data[i];
      }

      return results;
  }

  /**
   * @brief Print results of infer
   * @param results of the infer results
   * @param result_size of the struct of classification results
   * @param img_path image path
   * @return none
   */
  void print_infer_result(struct infer_result* results, size_t result_size, const char* img_path) {
      printf("\nImage %s\n", img_path);
      printf("\nclassid probability\n");
      printf("------- -----------\n");
      for (size_t i = 0; i < result_size; ++i) {
          printf("%zu       %f\n", results[i].class_id, results[i].probability);
      }
  }

  void print_model_input_output_info(ov_model_t* model) {
      char* friendly_name = NULL;
      ov_model_get_friendly_name(model, &friendly_name);
      printf("[INFO] model name: %s \n", friendly_name);
      ov_free(friendly_name);
  }

  /**
   * @brief Check image has supported width and height
   * @param string image size in WIDTHxHEIGHT format
   * @param pointer to image width
   * @param pointer to image height
   * @return bool status True(success) or False(fail)
   */

  bool is_supported_image_size(const char* size_str, size_t* width, size_t* height) {
      const char* _size = size_str;
      size_t _width = 0, _height = 0;
      while (_size && *_size != 'x' && *_size != '\0') {
          if ((*_size <= '9') && (*_size >= '0')) {
              _width = (_width * 10) + (*_size - '0');
              _size++;
          } else {
              goto err;
          }
      }

      if (_size)
          _size++;

      while (_size && *_size != '\0') {
          if ((*_size <= '9') && (*_size >= '0')) {
              _height = (_height * 10) + (*_size - '0');
              _size++;
          } else {
              goto err;
          }
      }

      if (_width > 0 && _height > 0) {
          if (_width % 2 == 0 && _height % 2 == 0) {
              *width = _width;
              *height = _height;
              return true;
          } else {
              printf("Unsupported image size, width and height must be even numbers \n");
              return false;
          }
      } else {
          goto err;
      }
  err:
      printf("Incorrect format of image size parameter, expected WIDTHxHEIGHT, "
             "actual: %s\n",
             size_str);
      return false;
  }

  size_t read_image_from_file(const char* img_path, unsigned char* img_data, size_t size) {
      FILE* fp = fopen(img_path, "rb");
      size_t read_size = 0;

      if (fp) {
          fseek(fp, 0, SEEK_END);
          if ((size_t)ftell(fp) >= size) {
              fseek(fp, 0, SEEK_SET);
              read_size = fread(img_data, 1, size, fp);
          }
          fclose(fp);
      }

      return read_size;
  }

  #define CHECK_STATUS(return_status)                                                      \
      if (return_status != OK) {                                                           \
          fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
          goto err;                                                                        \
      }

  int main(int argc, char** argv) {
      // -------- Check input parameters --------
      if (argc != 5) {
          printf("Usage : ./hello_nv12_input_classification_c <path_to_model> <path_to_image> "
                 "<WIDTHxHEIGHT> <device_name>\n");
          return EXIT_FAILURE;
      }

      size_t input_width = 0, input_height = 0, img_size = 0;
      if (!is_supported_image_size(argv[3], &input_width, &input_height)) {
          fprintf(stderr, "ERROR is_supported_image_size, line %d\n", __LINE__);
          return EXIT_FAILURE;
      }
      unsigned char* img_data = NULL;
      ov_core_t* core = NULL;
      ov_model_t* model = NULL;
      ov_tensor_t* tensor = NULL;
      ov_preprocess_prepostprocessor_t* preprocess = NULL;
      ov_preprocess_input_info_t* input_info = NULL;
      ov_model_t* new_model = NULL;
      ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
      ov_preprocess_preprocess_steps_t* input_process = NULL;
      ov_preprocess_input_model_info_t* p_input_model = NULL;
      ov_compiled_model_t* compiled_model = NULL;
      ov_infer_request_t* infer_request = NULL;
      ov_tensor_t* output_tensor = NULL;
      struct infer_result* results = NULL;
      char* input_tensor_name = NULL;
      char* output_tensor_name = NULL;
      ov_output_const_port_t* input_port = NULL;
      ov_output_const_port_t* output_port = NULL;
      ov_layout_t* model_layout = NULL;
      ov_shape_t input_shape = {.rank = 0, .dims = NULL};

      // -------- Get OpenVINO runtime version --------
      ov_version_t version = {.description = NULL, .buildNumber = NULL};
      CHECK_STATUS(ov_get_openvino_version(&version));
      printf("---- OpenVINO INFO----\n");
      printf("description : %s \n", version.description);
      printf("build number: %s \n", version.buildNumber);
      ov_version_free(&version);

      // -------- Parsing and validation of input arguments --------
      const char* input_model = argv[1];
      char input_image_path[PATH_MAX];
      if (!sanitize_path(argv[2], input_image_path, sizeof(input_image_path))) {
          goto err;
      }
      const char* device_name = argv[4];

      // -------- Step 1. Initialize OpenVINO Runtime Core --------
      CHECK_STATUS(ov_core_create(&core));

      // -------- Step 2. Read a model --------
      printf("[INFO] Loading model files: %s\n", input_model);
      CHECK_STATUS(ov_core_read_model(core, input_model, NULL, &model));
      print_model_input_output_info(model);

      CHECK_STATUS(ov_model_const_output(model, &output_port));

      CHECK_STATUS(ov_model_const_input(model, &input_port));

      CHECK_STATUS(ov_port_get_any_name(input_port, &input_tensor_name));
      CHECK_STATUS(ov_port_get_any_name(output_port, &output_tensor_name));

      // -------- Step 3. Configure preprocessing  --------
      CHECK_STATUS(ov_preprocess_prepostprocessor_create(model, &preprocess));

      // 1) Select input with 'input_tensor_name' tensor name
      CHECK_STATUS(ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, input_tensor_name, &input_info));

      // 2) Set input type
      // - as 'u8' precision
      // - set color format to NV12 (single plane)
      // - static spatial dimensions for resize preprocessing operation
      CHECK_STATUS(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
      CHECK_STATUS(ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, U8));
      CHECK_STATUS(ov_preprocess_input_tensor_info_set_color_format(input_tensor_info, NV12_SINGLE_PLANE));
      CHECK_STATUS(
          ov_preprocess_input_tensor_info_set_spatial_static_shape(input_tensor_info, input_height, input_width));

      // 3) Pre-processing steps:
      //    a) Convert to 'float'. This is to have color conversion more accurate
      //    b) Convert to BGR: Assumes that model accepts images in BGR format. For RGB, change it manually
      //    c) Resize image from tensor's dimensions to model ones
      CHECK_STATUS(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
      CHECK_STATUS(ov_preprocess_preprocess_steps_convert_element_type(input_process, F32));
      CHECK_STATUS(ov_preprocess_preprocess_steps_convert_color(input_process, BGR));
      CHECK_STATUS(ov_preprocess_preprocess_steps_resize(input_process, RESIZE_LINEAR));

      // 4) Set model data layout (Assuming model accepts images in NCHW layout)
      CHECK_STATUS(ov_preprocess_input_info_get_model_info(input_info, &p_input_model));

      const char* model_layout_desc = "NCHW";
      CHECK_STATUS(ov_layout_create(model_layout_desc, &model_layout));
      CHECK_STATUS(ov_preprocess_input_model_info_set_layout(p_input_model, model_layout));

      // 5) Apply preprocessing to an input with 'input_tensor_name' name of loaded model
      CHECK_STATUS(ov_preprocess_prepostprocessor_build(preprocess, &new_model));

      // -------- Step 4. Loading a model to the device --------
      CHECK_STATUS(ov_core_compile_model(core, new_model, device_name, 0, &compiled_model));

      // -------- Step 5. Create an infer request --------
      CHECK_STATUS(ov_compiled_model_create_infer_request(compiled_model, &infer_request));

      // -------- Step 6. Prepare input data  --------
      img_size = input_width * (input_height * 3 / 2);
      if (!img_size) {
          fprintf(stderr, "[ERROR] Invalid Image size, line %d\n", __LINE__);
          goto err;
      }
      img_data = (unsigned char*)calloc(img_size, sizeof(unsigned char));
      if (!img_data) {
          fprintf(stderr, "[ERROR] calloc returned NULL, line %d\n", __LINE__);
          goto err;
      }
      if (img_size != read_image_from_file(input_image_path, img_data, img_size)) {
          fprintf(stderr, "[ERROR] Image dimensions not match with NV12 file size, line %d\n", __LINE__);
          goto err;
      }
      ov_element_type_e input_type = U8;
      size_t batch = 1;
      int64_t dims[4] = {batch, input_height * 3 / 2, input_width, 1};
      ov_shape_create(4, dims, &input_shape);
      CHECK_STATUS(ov_tensor_create_from_host_ptr(input_type, input_shape, img_data, &tensor));

      // -------- Step 6. Set input tensor  --------
      // Set the input tensor by tensor name to the InferRequest
      CHECK_STATUS(ov_infer_request_set_tensor(infer_request, input_tensor_name, tensor));

      // -------- Step 7. Do inference --------
      // Running the request synchronously
      CHECK_STATUS(ov_infer_request_infer(infer_request));

      // -------- Step 8. Process output --------
      CHECK_STATUS(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
      // Print classification results
      size_t results_num = 0;
      results = tensor_to_infer_result(output_tensor, &results_num);
      if (!results) {
          goto err;
      }
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
      free(img_data);
      ov_shape_free(&input_shape);
      ov_free(input_tensor_name);
      ov_free(output_tensor_name);
      ov_output_const_port_free(output_port);
      ov_output_const_port_free(input_port);
      if (output_tensor)
          ov_tensor_free(output_tensor);
      if (infer_request)
          ov_infer_request_free(infer_request);
      if (compiled_model)
          ov_compiled_model_free(compiled_model);
      if (p_input_model)
          ov_preprocess_input_model_info_free(p_input_model);
      if (input_process)
          ov_preprocess_preprocess_steps_free(input_process);
      if (model_layout)
          ov_layout_free(model_layout);
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
  <TabItem value="cpp" label="C++">
  ```console
  hello_nv12_input_classification <path_to_model> <path_to_image> <image_size> <device_name>
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```console
  hello_nv12_input_classification_c <path_to_model> <path_to_image> <device_name>
  ```
  </TabItem>
</Tabs>

To run the sample, you need to specify a model and an image:

- You can get a model specific for your inference task from one of model
  repositories, such as TensorFlow Zoo, HuggingFace, or TensorFlow Hub.
- You can use images from the media files collection available at
  [the storage](https://storage.openvinotoolkit.org/data/test_data).

The sample accepts an uncompressed image in the NV12 color format. To run the
sample, you need to convert your BGR/RGB image to NV12. To do this, you can use
one of the widely available tools such as FFmpeg or GStreamer. Using FFmpeg and
the following command, you can convert an ordinary image to an uncompressed NV12 image:

```bash
ffmpeg -i cat.jpg -pix_fmt nv12 cat.yuv
```

:::note
- Because the sample reads raw image files, you should provide a correct image
  size along with the image path. The sample expects the logical size of the
  image, not the buffer size. For example, for 640x480 BGR/RGB image the
  corresponding NV12 logical image size is also 640x480, whereas the buffer
  size is 640x720.
- By default, this sample expects that model input has BGR channels order. If
  you trained your model to work with RGB order, you need to reconvert your
  model using model conversion API with `reverse_input_channels` argument
  specified. For more information about the argument, refer to the
  **Color Conversion** section of [Preprocessing API](../../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details/index.md).
- Before running the sample with a trained model, make sure the model is
  converted to the intermediate representation (IR) format (\*.xml + \*.bin)
  using the [model conversion API](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md).
- The sample accepts models in ONNX format (.onnx) that do not require preprocessing.
:::

### Example

1. Download a pre-trained model.
2. You can convert it by using:
   
   ```console
   ovc ./models/alexnet
   ```
3. Perform inference of an NV12 image, using a model on a `CPU`, for example:
   
   <Tabs>
     <TabItem value="cpp" label="C++">
     ```console
     hello_nv12_input_classification ./models/alexnet.xml ./images/cat.yuv 300x300 CPU
     ```
     </TabItem>
     <TabItem value="c" label="C">
     ```console
     hello_nv12_input_classification_c ./models/alexnet.xml ./images/cat.yuv 300x300 CPU
     ```
     </TabItem>
   </Tabs>

## Sample Output

<Tabs>
  <TabItem value="cpp" label="C++">
  The application outputs top-10 inference results.

  ```console
  [ INFO ] OpenVINO Runtime version ......... <version>
  [ INFO ] Build ........... <build>
  [ INFO ]
  [ INFO ] Loading model files: \models\alexnet.xml
  [ INFO ] model name: AlexNet
  [ INFO ]     inputs
  [ INFO ]         input name: data
  [ INFO ]         input type: f32
  [ INFO ]         input shape: {1, 3, 227, 227}
  [ INFO ]     outputs
  [ INFO ]         output name: prob
  [ INFO ]         output type: f32
  [ INFO ]         output shape: {1, 1000}

  Top 10 results:

  Image \images\car.yuv

  classid probability
  ------- -----------
  656     0.6668988
  654     0.1125269
  581     0.0679280
  874     0.0340229
  436     0.0257744
  817     0.0169367
  675     0.0110199
  511     0.0106134
  569     0.0083373
  717     0.0061734
  ```
  </TabItem>
  <TabItem value="c" label="C">
  The application outputs top-10 inference results.

  ```console
  Top 10 results:

  Image ./cat.yuv

  classid probability
  ------- -----------
  435       0.091733
  876       0.081725
  999       0.069305
  587       0.043726
  666       0.038957
  419       0.032892
  285       0.030309
  700       0.029941
  696       0.021628
  855       0.020339

  This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [Integrate the OpenVINO™ Runtime with Your Application](../../../openvino-workflow/running-inference/index.md)
- [Get Started with Samples](get-started-demos.md)
- [Using OpenVINO Samples](index.md)
- [Convert a Model](../../../openvino-workflow/model-preparation/convert-model-to-ir/index.md)
- [API Reference](https://docs.openvino.ai/2026/api/api_reference.html)
- [Hello NV12 Input Classification C++ Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/hello_nv12_input_classification/README.md)
- [Hello NV12 Input Classification C Sample on Github](https://github.com/openvinotoolkit/openvino/blob/master/samples/c/hello_nv12_input_classification/README.md)
