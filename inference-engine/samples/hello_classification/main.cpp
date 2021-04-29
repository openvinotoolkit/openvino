// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <samples/classification_results.h>

#include <inference_engine.hpp>
#include <iterator>
#include <memory>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include <string>
#include <vector>

using namespace InferenceEngine;

/**
 * @brief Define names based depends on Unicode path support
 */
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    #define tcout                  std::wcout
    #define file_name_t            std::wstring
    #define imread_t               imreadW
    #define ClassificationResult_t ClassificationResultW
#else
    #define tcout                  std::cout
    #define file_name_t            std::string
    #define imread_t               cv::imread
    #define ClassificationResult_t ClassificationResult
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
            std::copy(std::istreambuf_iterator<char>(input_image_stream), std::istreambuf_iterator<char>(), std::back_inserter(buffer));
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
        // ------------------------------ Parsing and validation of input arguments
        // ---------------------------------
        if (argc != 4) {
            tcout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name>" << std::endl;
            return EXIT_FAILURE;
        }

        const file_name_t input_model {argv[1]};
        const file_name_t input_image_path {argv[2]};
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::string device_name = simpleConvert(argv[3]);
#else
        const std::string device_name {argv[3]};
#endif
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        Core ie;
        // -----------------------------------------------------------------------------------------------------

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        CNNNetwork network = ie.ReadNetwork(input_model);
        if (network.getOutputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 output only");
        if (network.getInputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 input only");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 3. Configure input & output
        // ---------------------------------------------
        // --------------------------- Prepare input blobs
        // -----------------------------------------------------
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        std::string input_name = network.getInputsInfo().begin()->first;

        /* Mark input as resizable by setting of a resize algorithm.
         * In this case we will be able to set an input blob of any shape to an
         * infer request. Resize and layout conversions are executed automatically
         * during inference */
        input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_info->setLayout(Layout::NHWC);
        input_info->setPrecision(Precision::U8);

        // --------------------------- Prepare output blobs
        // ----------------------------------------------------
        if (network.getOutputsInfo().empty()) {
            std::cerr << "Network outputs info is empty" << std::endl;
            return EXIT_FAILURE;
        }
        DataPtr output_info = network.getOutputsInfo().begin()->second;
        std::string output_name = network.getOutputsInfo().begin()->first;

        output_info->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Loading a model to the device
        // ------------------------------------------
        ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create an infer request
        // -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------
        /* Read input image to a blob and set it to an infer request without resize
         * and layout conversions. */
        cv::Mat image = imread_t(input_image_path);
        Blob::Ptr imgBlob = wrapMat2Blob(image);     // just wrap Mat data by Blob::Ptr
                                                     // without allocating of new memory
        infer_request.SetBlob(input_name, imgBlob);  // infer_request accepts input blob of any size
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 7. Do inference
        // --------------------------------------------------------
        /* Running the request synchronously */
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 8. Process output
        // ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);
        // Print classification results
        ClassificationResult_t classificationResult(output, {input_image_path});
        classificationResult.print();
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
