// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <samples/classification_results.h>
#include <sys/stat.h>

#include <fstream>
#include <inference_engine.hpp>
#include <iostream>
#include <memory>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
    #include <samples/os/windows/w_dirent.h>
#else
    #include <dirent.h>
#endif

using namespace InferenceEngine;

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
 * @brief Read input image to blob
 * @param ref to input image data
 * @param width input image
 * @param height input image
 * @return blob point to hold the NV12 input data
 */
std::vector<Blob::Ptr> readInputBlobs(std::vector<UString>& data, size_t width, size_t height) {
    // read image with size converted to NV12 data size: height(NV12) = 3 / 2 *
    // logical height

    // Create tensor descriptors for Y and UV blobs
    const InferenceEngine::TensorDesc y_plane_desc(InferenceEngine::Precision::U8, {1, 1, height, width}, InferenceEngine::Layout::NHWC);
    const InferenceEngine::TensorDesc uv_plane_desc(InferenceEngine::Precision::U8, {1, 2, height / 2, width / 2}, InferenceEngine::Layout::NHWC);
    const size_t offset = width * height;

    std::vector<Blob::Ptr> blobs;
    for (auto& buf : data) {
        // --------------------------- Create a blob to hold the NV12 input data
        // -------------------------------
        auto ptr = &buf[0];

        // Create blob for Y plane from raw data
        Blob::Ptr y_blob = make_shared_blob<uint8_t>(y_plane_desc, ptr);
        // Create blob for UV plane from raw data
        Blob::Ptr uv_blob = make_shared_blob<uint8_t>(uv_plane_desc, ptr + offset);
        // Create NV12Blob from Y and UV blobs
        blobs.emplace_back(make_shared_blob<NV12Blob>(y_blob, uv_blob));
    }

    return blobs;
}

/**
 * @brief Check supported batched blob for device
 * @param IE core object
 * @param string device name
 * @return True(success) or False(fail)
 */
bool isBatchedBlobSupported(const Core& ie, const std::string& device_name) {
    const std::vector<std::string> supported_metrics = ie.GetMetric(device_name, METRIC_KEY(SUPPORTED_METRICS));

    if (std::find(supported_metrics.begin(), supported_metrics.end(), METRIC_KEY(OPTIMIZATION_CAPABILITIES)) == supported_metrics.end()) {
        return false;
    }

    const std::vector<std::string> optimization_caps = ie.GetMetric(device_name, METRIC_KEY(OPTIMIZATION_CAPABILITIES));

    return std::find(optimization_caps.begin(), optimization_caps.end(), METRIC_VALUE(BATCHED_BLOB)) != optimization_caps.end();
}

/**
 * @brief The entry point of the Inference Engine sample application
 */
int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Parsing and validation input
        // arguments------------------------------
        if (argc != 5) {
            std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image(s)> <image_size> <device_name>" << std::endl;
            return EXIT_FAILURE;
        }

        const std::string input_model {argv[1]};
        const std::string input_image_path {argv[2]};
        size_t input_width = 0, input_height = 0;
        std::tie(input_width, input_height) = parseImageSize(argv[3]);
        const std::string device_name {argv[4]};
        // -----------------------------------------------------------------------------------------------------

        // ------------------------------ Read image names
        // -----------------------------------------------------
        auto image_names = readInputFileNames(input_image_path);

        if (image_names.empty()) {
            throw std::invalid_argument("images not found");
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // ------------------------------------------------
        Core ie;
        // -----------------------------------------------------------------------------------------------------

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        CNNNetwork network = ie.ReadNetwork(input_model);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Reshape model
        // -------------------------------------------------
        size_t netInputSize = isBatchedBlobSupported(ie, device_name) ? image_names.size() : 1;
        ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
        for (auto& shape : inputShapes) {
            auto& dims = shape.second;
            if (dims.empty()) {
                throw std::runtime_error("Network's input shapes have empty dimensions");
            }
            dims[0] = netInputSize;
        }
        network.reshape(inputShapes);
        size_t batchSize = network.getBatchSize();
        std::cout << "Batch size is " << batchSize << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 3. Configure input and output
        // -------------------------------------------
        // --------------------------- Prepare input blobs
        // -----------------------------------------------------
        if (network.getInputsInfo().empty()) {
            std::cerr << "Network inputs info is empty" << std::endl;
            return EXIT_FAILURE;
        }
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        std::string input_name = network.getInputsInfo().begin()->first;

        input_info->setLayout(Layout::NCHW);
        input_info->setPrecision(Precision::U8);
        // set input resize algorithm to enable input autoresize
        input_info->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        // set input color format to ColorFormat::NV12 to enable automatic input
        // color format pre-processing
        input_info->getPreProcess().setColorFormat(ColorFormat::NV12);

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
        // ----------------------------------------
        ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create an infer request
        // ----------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------
        auto image_bufs = readImagesDataFromFiles(image_names, input_width * (input_height * 3 / 2));

        auto inputs = readInputBlobs(image_bufs, input_width, input_height);

        // If batch_size > 1 => batched blob supported => replace all inputs by a
        // BatchedBlob
        if (netInputSize > 1) {
            assert(netInputSize == inputs.size());
            std::cout << "Infer using BatchedBlob of NV12 images." << std::endl;
            Blob::Ptr batched_input = make_shared_blob<BatchedBlob>(inputs);
            inputs = {batched_input};
        }

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
            // --------------------------- Set the input blob to the InferRequest
            // ------------------------------
            infer_request.SetBlob(input_name, input);
            // -------------------------------------------------------------------------------------------------

            // --------------------------- Step 7. Do inference
            // -----------------------------------------------------
            /* Running the request synchronously */
            infer_request.Infer();
            // -------------------------------------------------------------------------------------------------

            // --------------------------- Step 8. Process output
            // ---------------------------------------------------
            Blob::Ptr output = infer_request.GetBlob(output_name);

            // Print classification results
            const auto names_offset = image_names.begin() + netInputSize * i;
            std::vector<std::string> names(names_offset, names_offset + netInputSize);

            ClassificationResult classificationResult(output, names, netInputSize, 10, labels);
            classificationResult.print();
            // -------------------------------------------------------------------------------------------------
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
