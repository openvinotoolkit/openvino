// Copyright (C) 2018-2025 Intel Corporation
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
