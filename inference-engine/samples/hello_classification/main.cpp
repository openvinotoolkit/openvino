// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <samples/slog.hpp>
#include <string>
#include <vector>

#include "openvino/core/layout.hpp"
#include "openvino/openvino.hpp"
#include "samples/classification_results.h"
#include "samples/common.hpp"
#include "samples/ocv_common.hpp"

using namespace ov::preprocess;

/**
 * @brief Define names based depends on Unicode path support
 */
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#    define tcout                  std::wcout
#    define file_name_t            std::wstring
#    define imread_t               imreadW
#    define ClassificationResult_t ClassificationResultW
#else
#    define tcout                  std::cout
#    define file_name_t            std::string
#    define imread_t               cv::imread
#    define ClassificationResult_t ClassificationResult
#endif

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
/**
 * @brief Realization cv::imread with support Unicode paths
 */
cv::Mat imreadW(std::wstring input_image_path) {
    cv::Mat image;
    std::ifstream input_image_stream;
    input_image_stream.open(input_image_path.c_str(), std::iostream::binary | std::ios_base::ate | std::ios_base::in);
    if (input_image_stream.is_open()) {
        if (input_image_stream.good()) {
            input_image_stream.seekg(0, std::ios::end);
            std::size_t file_size = input_image_stream.tellg();
            input_image_stream.seekg(0, std::ios::beg);
            std::vector<char> buffer(0);
            std::copy(std::istreambuf_iterator<char>(input_image_stream),
                      std::istreambuf_iterator<char>(),
                      std::back_inserter(buffer));
            image = cv::imdecode(cv::Mat(1, file_size, CV_8UC1, &buffer[0]), cv::IMREAD_COLOR);
        } else {
            tcout << "Input file '" << input_image_path << "' processing error" << std::endl;
        }
        input_image_stream.close();
    } else {
        tcout << "Unable to read input file '" << input_image_path << "'" << std::endl;
    }
    return image;
}

/**
 * @brief Convert wstring to string
 * @param ref on wstring
 * @return string
 */
std::string simpleConvert(const std::wstring& wstr) {
    std::string str;
    for (auto&& wc : wstr)
        str += static_cast<char>(wc);
    return str;
}

/**
 * @brief Main with support Unicode paths, wide strings
 */
int wmain(int argc, wchar_t* argv[]) {
#else

int main(int argc, char* argv[]) {
#endif
    try {
        // -------- Get OpenVINO Runtime version --------
        slog::info << "OpenVINO runtime: " << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 4) {
            tcout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name>" << std::endl;
            return EXIT_FAILURE;
        }

        const file_name_t input_model{argv[1]};
        const file_name_t input_image_path{argv[2]};
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::string device_name = simpleConvert(argv[3]);
#else
        const std::string device_name{argv[3]};
#endif

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::runtime::Core core;

        // -------- Step 2. Read a model --------
        auto model = core.read_model(input_model);

        OPENVINO_ASSERT(model->get_parameters().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->get_results().size() == 1, "Sample supports models with 1 output only");

        // -------- Step 3. Initialize inference engine core

        // Read input image to a tensor and set it to an infer request
        // without resize and layout conversions
        cv::Mat image = imread_t(input_image_path);
        // just wrap Mat data by ov::runtime::Tensor without allocating of new memory
        ov::runtime::Tensor input_tensor = wrapMat2Tensor(image);
        const ov::Shape tensor_shape = input_tensor.get_shape();

        // -------- Step 4. Apply preprocessing --------
        const ov::Layout tensor_layout{"NHWC"};

        // clang-format off
        model = PrePostProcessor().
            // 1) InputInfo() with no args assumes a model has a single input
            input(InputInfo().
                // 2) Set input tensor information:
                // - precision of tensor is supposed to be 'u8'
                // - layout of data is 'NHWC'
                // - set static spatial dimensions to input tensor to resize from
                tensor(InputTensorInfo().
                    set_element_type(ov::element::u8).
                    set_spatial_static_shape(
                        tensor_shape[ov::layout::height_idx(tensor_layout)],
                        tensor_shape[ov::layout::width_idx(tensor_layout)]).
                    set_layout(tensor_layout)).
                // 3) Adding explicit preprocessing steps:
                // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
                // - apply linear resize from tensor spatial dims to model spatial dims
                preprocess(PreProcessSteps().
                    convert_element_type(ov::element::f32). // WA for CPU plugin
                    convert_layout("NCHW"). // WA for CPU plugin
                    resize(ResizeAlgorithm::RESIZE_LINEAR)).
                // 4) Here we suppose model has 'NCHW' layout for input
                network(InputNetworkInfo().
                    set_layout("NCHW"))).
            output(OutputInfo().
                // 5) Set output tensor information:
                // - precision of tensor is supposed to be 'f32'
                tensor(OutputTensorInfo().
                    set_element_type(ov::element::f32))).
        // 6) Apply preprocessing modifing the original 'model'
        build(model);
        // clang-format on

        // -------- Step 5. Loading a model to the device --------
        ov::runtime::ExecutableNetwork executable_network = core.compile_model(model, device_name);

        // -------- Step 6. Create an infer request --------
        ov::runtime::InferRequest infer_request = executable_network.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // -------- Step 7. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output
        ov::runtime::Tensor output_tensor = infer_request.get_output_tensor();

        // Print classification results
        ClassificationResult_t classification_result(output_tensor, {input_image_path});
        classification_result.show();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool"
              << std::endl;
    return EXIT_SUCCESS;
}
