// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <samples/classification_results.h>
#include <sys/stat.h>

#include <fstream>
#include <inference_engine.hpp>
#include <iostream>
#include <memory>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/core.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#    include <samples/os/windows/w_dirent.h>
#else
#    include <dirent.h>
#endif

using namespace InferenceEngine;

// TODO: avoid conversion to legacy API
inline Precision convertPrecision(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::undefined:
        return Precision(Precision::UNSPECIFIED);
    case ov::element::f16:
        return Precision(Precision::FP16);
    case ov::element::f32:
        return Precision(Precision::FP32);
    case ov::element::f64:
        return Precision(Precision::FP64);
    case ov::element::bf16:
        return Precision(Precision::BF16);
    case ov::element::i4:
        return Precision(Precision::I4);
    case ov::element::i8:
        return Precision(Precision::I8);
    case ov::element::i16:
        return Precision(Precision::I16);
    case ov::element::i32:
        return Precision(Precision::I32);
    case ov::element::i64:
        return Precision(Precision::I64);
    case ov::element::u4:
        return Precision(Precision::U4);
    case ov::element::u8:
        return Precision(Precision::U8);
    case ov::element::u16:
        return Precision(Precision::U16);
    case ov::element::u32:
        return Precision(Precision::U32);
    case ov::element::u64:
        return Precision(Precision::U64);
    case ov::element::u1:
        return Precision(Precision::BIN);
    case ov::element::boolean:
        return Precision(Precision::BOOL);
    case ov::element::dynamic:
        return Precision(Precision::UNSPECIFIED);
    default:
        IE_THROW() << "Incorrect precision " << precision.get_type_name() << "!";
        return {};
    }
}

/**
 * \brief Parse image size provided as string in format WIDTHxHEIGHT
 * @param string of image size in WIDTHxHEIGHT format
 * @return parsed width and height
 */
std::pair<size_t, size_t> parseImageSize(const std::string& size_string) {
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

// Comparing to samples/args_helper.hpp, this version filters files by ".yuv"
// extension
/**
 * @brief This function checks input args and existence of specified files in a
 * given folder
 * @param path path to a file to be checked for existence
 * @return files updated vector of verified input files
 */
std::vector<std::string> readInputFileNames(const std::string& path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) != 0) {
        slog::warn << "File " << path << " cannot be opened!" << slog::endl;
        return {};
    }

    std::vector<std::string> files;

    if (S_ISDIR(sb.st_mode)) {
        DIR* dp = opendir(path.c_str());
        if (dp == nullptr) {
            slog::warn << "Directory " << path << " cannot be opened!" << slog::endl;
            return {};
        }

        for (struct dirent* ep = readdir(dp); ep != nullptr; ep = readdir(dp)) {
            std::string fileName = ep->d_name;
            if (fileName == "." || fileName == ".." || fileName.substr(fileName.size() - 4) != ".yuv")
                continue;
            files.push_back(path + "/" + ep->d_name);
        }
        closedir(dp);
    } else {
        files.push_back(path);
    }

    size_t max_files = 20;
    if (files.size() < max_files) {
        slog::info << "Files were added: " << files.size() << slog::endl;
        for (std::string filePath : files) {
            slog::info << "    " << filePath << slog::endl;
        }
    } else {
        slog::info << "Files were added: " << files.size() << ". Too many to display each of them." << slog::endl;
    }

    return files;
}

using UString = std::basic_string<uint8_t>;

/**
 * \brief Read image data from file
 * @param vector files paths
 * @param size of file paths vector
 * @return buffers containing the images data
 */
std::vector<UString> readImagesDataFromFiles(const std::vector<std::string>& files, size_t size) {
    std::vector<UString> result;

    for (const auto& image_path : files) {
        std::ifstream file(image_path, std::ios_base::ate | std::ios_base::binary);
        if (!file.good() || !file.is_open()) {
            std::stringstream err;
            err << "Cannot access input image file. File path: " << image_path;
            throw std::runtime_error(err.str());
        }

        const size_t file_size = file.tellg();
        if (file_size < size) {
            std::stringstream err;
            err << "Invalid read size provided. File size: " << file_size << ", to read: " << size;
            throw std::runtime_error(err.str());
        }
        file.seekg(0);

        UString data(size, 0);
        file.read(reinterpret_cast<char*>(&data[0]), size);
        result.push_back(std::move(data));
    }
    return result;
}

/**
 * @brief Read input NV12 image to tensor
 * @param ref to input image data
 * @param width input image width
 * @param height input image height (actual NV12 buffer height is 1.5x bigger than 'height')
 * @return Tensor holding the NV12 input data
 */
ov::runtime::TensorVector readInputTensors(std::vector<UString>& data, size_t width, size_t height) {
    ov::runtime::TensorVector tensors;
    for (auto& buf : data) {
        // ---------- Create a tensor to hold the NV12 input data -------------------
        auto ptr = &buf[0];

        // Create tensor for Y plane from raw data
        ov::runtime::Tensor yuv = ov::runtime::Tensor(ov::element::u8, {1, height * 3 / 2, width, 1}, ptr);
        tensors.emplace_back(yuv);
    }

    return tensors;
}

/**
 * @brief The entry point of the Inference Engine sample application
 */
int main(int argc, char* argv[]) {
    try {
        // ------------ Parsing and validation input arguments--------
        if (argc != 5) {
            std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image(s)> <image_size> <device_name>"
                      << std::endl;
            return EXIT_FAILURE;
        }

        const std::string input_model{argv[1]};
        const std::string input_image_path{argv[2]};
        size_t input_width = 0, input_height = 0;
        std::tie(input_width, input_height) = parseImageSize(argv[3]);
        const std::string device_name{argv[4]};
        // ------------------------------------------------------------

        // --------------------- Read image names ---------------------
        auto image_names = readInputFileNames(input_image_path);

        size_t netInputSize = 1;
        if (image_names.empty()) {
            throw std::invalid_argument("images not found");
        }
        // -----------------------------------------------------------

        // -------- Step 1. Initialize inference engine core ---------
        ov::runtime::Core ie;
        // -----------------------------------------------------------

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        auto function = ie.read_model(input_model);
        std::string input_name = function->get_parameters()[0]->get_friendly_name();
        std::string output_name = function->get_result()->get_input_source_output(0).get_node()->get_friendly_name();

        // --------- Step 3. Add preprocessing  ---------
        // 1) Set input type as 'u8' precision and set color format to NV12 (single plane)
        ov::preprocess::PrePostProcessor p;
        auto tensor = ov::preprocess::InputTensorInfo()
                          .set_element_type(ov::element::u8)
                          .set_color_format(ov::preprocess::ColorFormat::NV12_SINGLE_PLANE)
                          .set_spatial_static_shape(input_height, input_width);
        // 2) Pre-processing steps:
        //    a) Convert to 'float'. This is to have color conversion more accurate
        //    b) Convert to RGB: Assumes that model accepts images in RGB format. For BGR, change it manually
        //    c) Convert layout to network's one. It is done before 'resize' in this sample, as there can be plugin
        //       limitation with support of resize in NHWC format
        //    d) Resize image from tensor's dimensions to network ones
        auto steps = ov::preprocess::PreProcessSteps()
                         .convert_element_type(ov::element::f32)
                         .convert_color(ov::preprocess::ColorFormat::RGB)
                         .convert_layout()
                         .resize(ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC);
        // 3) Set network data layout (Assuming model accepts images in NCHW layout)
        auto netInfo = ov::preprocess::InputNetworkInfo().set_layout("NCHW");
        // 4) Apply preprocessing to a first input of loaded function
        function = p.input(ov::preprocess::InputInfo(0)
                               .tensor(std::move(tensor))
                               .preprocess(std::move(steps))
                               .network(std::move(netInfo)))
                       .build(function);

        // -------------- Step 4. Loading a model to the device ---------------
        ov::runtime::ExecutableNetwork executable_network = ie.compile_model(function, device_name);
        // --------------------------------------------------------------------

        // -------------- Step 5. Create an infer request ---------------------
        ov::runtime::InferRequest infer_request = executable_network.create_infer_request();
        // --------------------------------------------------------------------

        // -------------- Step 6. Prepare input -------------------------------
        auto image_bufs = readImagesDataFromFiles(image_names, input_width * (input_height * 3 / 2));

        auto inputs = readInputTensors(image_bufs, input_width, input_height);

        /** Read labels from file (e.x. AlexNet.labels) **/
        std::string labelFileName = fileNameNoExt(input_model) + ".labels";
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

        for (size_t i = 0; i < inputs.size(); i++) {
            const auto& input = inputs[i];
            // ----------- Set the input tensor to the InferRequest ----
            infer_request.set_tensor(input_name, input);
            // -------------------------------------------------------

            // ------------ Step 7. Do inference ---------------------
            /* Running the request synchronously */
            infer_request.infer();
            // -------------------------------------------------------

            // ------------ Step 8. Process output -------------------
            ov::runtime::Tensor output = infer_request.get_tensor(output_name);

            // Print classification results
            const auto names_offset = image_names.begin() + netInputSize * i;
            std::vector<std::string> names(names_offset, names_offset + netInputSize);

            // TODO: avoid conversion to legacy types (Blob, etc)
            InferenceEngine::TensorDesc desc(convertPrecision(output.get_element_type()), InferenceEngine::Layout::ANY);
            desc.setDims(output.get_shape());
            InferenceEngine::Blob::Ptr out_blob = make_shared_blob(desc, output.data<float>());
            ClassificationResult classificationResult(out_blob, names, netInputSize, 10, labels);
            classificationResult.print();
            // -------------------------------------------------------
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool"
              << std::endl;
    return EXIT_SUCCESS;
}
