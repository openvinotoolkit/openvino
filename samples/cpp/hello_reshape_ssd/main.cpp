// Copyright (C) 2018-2025 Intel Corporation
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

        // 7) Apply preprocessing modifing the original 'model'
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
