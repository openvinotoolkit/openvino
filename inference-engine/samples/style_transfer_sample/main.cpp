// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>

#include <inference_engine.hpp>
#include <memory>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <vector>

#include "style_transfer_sample.h"

using namespace InferenceEngine;

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_m.empty()) {
        showUsage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_i.empty()) {
        showUsage();
        throw std::logic_error("Input is required but not set. Please set -i option.");
    }

    return true;
}

/**
 * @brief The entry point for inference engine deconvolution sample application
 * @file style_transfer_sample/main.cpp
 * @example style_transfer_sample/main.cpp
 */
int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Get Inference Engine version
        // ------------------------------------------------------
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        // ------------------------------ Parsing and validation of input arguments
        // ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty())
            throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        // ------------------------------ Get Available Devices
        // ------------------------------------------------------
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d) << std::endl;

        if (!FLAGS_l.empty()) {
            // Custom CPU extension is loaded as a shared library and passed as a
            // pointer to base extension
            IExtensionPtr extension_ptr = std::make_shared<Extension>(FLAGS_l);
            ie.AddExtension(extension_ptr);
            slog::info << "Custom Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty() && (FLAGS_d == "GPU" || FLAGS_d == "MYRIAD" || FLAGS_d == "HDDL")) {
            // Config for device plugin custom extension is loaded from an .xml
            // description
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            slog::info << "Config for " << FLAGS_d << " device plugin custom extension loaded: " << FLAGS_c << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        slog::info << "Loading network files:" << slog::endl << FLAGS_m << slog::endl;

        /** Read network model **/
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 3. Configure input & output
        // ---------------------------------------------

        // --------------------------- Prepare input blobs
        // -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());

        if (inputInfo.size() != 1)
            throw std::logic_error("Sample supports topologies only with 1 input");
        auto inputInfoItem = *inputInfo.begin();

        /** Iterate over all the input blobs **/
        std::vector<std::shared_ptr<uint8_t>> imagesData;

        /** Specifying the precision of input data.
         * This should be called before load of the network to the device **/
        inputInfoItem.second->setPrecision(Precision::FP32);

        /** Collect images data ptrs **/
        for (auto& i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(
                reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty())
            throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        slog::info << "Batch size is " << std::to_string(network.getBatchSize()) << slog::endl;

        // ------------------------------ Prepare output blobs
        // -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        // BlobMap outputBlobs;
        std::string firstOutputName;

        const float meanValues[] = {static_cast<const float>(FLAGS_mean_val_r), static_cast<const float>(FLAGS_mean_val_g),
                                    static_cast<const float>(FLAGS_mean_val_b)};

        for (auto& item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Loading model to the device
        // ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create infer request
        // -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------
        /** Iterate over all the input blobs **/
        for (const auto& item : inputInfo) {
            MemoryBlob::Ptr minput = as<MemoryBlob>(infer_request.GetBlob(item.first));
            if (!minput) {
                slog::err << "We expect input blob to be inherited from MemoryBlob, "
                          << "but by fact we were not able to cast it to MemoryBlob" << slog::endl;
                return 1;
            }
            // locked memory holder should be alive all time while access to its
            // buffer happens
            auto ilmHolder = minput->wmap();

            /** Filling input tensor with images. First b channel, then g and r
             * channels **/
            size_t num_channels = minput->getTensorDesc().getDims()[1];
            size_t image_size = minput->getTensorDesc().getDims()[3] * minput->getTensorDesc().getDims()[2];

            auto data = ilmHolder.as<PrecisionTrait<Precision::FP32>::value_type*>();
            if (data == nullptr)
                throw std::runtime_error("Input blob has not allocated buffer");
            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in
                         * bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid] =
                            imagesData.at(image_id).get()[pid * num_channels + ch] - meanValues[ch];
                    }
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 7. Do inference
        // ---------------------------------------------------------
        slog::info << "Start inference" << slog::endl;
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 8. Process output
        // -------------------------------------------------------
        MemoryBlob::CPtr moutput = as<MemoryBlob>(infer_request.GetBlob(firstOutputName));
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                   "but by fact we were not able to cast it to MemoryBlob");
        }
        // locked memory holder should be alive all time while access to its buffer
        // happens
        auto lmoHolder = moutput->rmap();
        const auto output_data = lmoHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

        size_t num_images = moutput->getTensorDesc().getDims()[0];
        size_t num_channels = moutput->getTensorDesc().getDims()[1];
        size_t H = moutput->getTensorDesc().getDims()[2];
        size_t W = moutput->getTensorDesc().getDims()[3];
        size_t nPixels = W * H;

        slog::info << "Output size [N,C,H,W]: " << num_images << ", " << num_channels << ", " << H << ", " << W << slog::endl;

        {
            std::vector<float> data_img(nPixels * num_channels);

            for (size_t n = 0; n < num_images; n++) {
                for (size_t i = 0; i < nPixels; i++) {
                    data_img[i * num_channels] = static_cast<float>(output_data[i + n * nPixels * num_channels] + meanValues[0]);
                    data_img[i * num_channels + 1] = static_cast<float>(output_data[(i + nPixels) + n * nPixels * num_channels] + meanValues[1]);
                    data_img[i * num_channels + 2] = static_cast<float>(output_data[(i + 2 * nPixels) + n * nPixels * num_channels] + meanValues[2]);

                    float temp = data_img[i * num_channels];
                    data_img[i * num_channels] = data_img[i * num_channels + 2];
                    data_img[i * num_channels + 2] = temp;

                    if (data_img[i * num_channels] < 0)
                        data_img[i * num_channels] = 0;
                    if (data_img[i * num_channels] > 255)
                        data_img[i * num_channels] = 255;

                    if (data_img[i * num_channels + 1] < 0)
                        data_img[i * num_channels + 1] = 0;
                    if (data_img[i * num_channels + 1] > 255)
                        data_img[i * num_channels + 1] = 255;

                    if (data_img[i * num_channels + 2] < 0)
                        data_img[i * num_channels + 2] = 0;
                    if (data_img[i * num_channels + 2] > 255)
                        data_img[i * num_channels + 2] = 255;
                }
                std::string out_img_name = std::string("out" + std::to_string(n + 1) + ".bmp");
                std::ofstream outFile;
                outFile.open(out_img_name.c_str(), std::ios_base::binary);
                if (!outFile.is_open()) {
                    throw new std::runtime_error("Cannot create " + out_img_name);
                }
                std::vector<unsigned char> data_img2;
                for (float i : data_img) {
                    data_img2.push_back(static_cast<unsigned char>(i));
                }
                writeOutputBmp(data_img2.data(), H, W, outFile);
                outFile.close();
                slog::info << "Image " << out_img_name << " created!" << slog::endl;
            }
        }
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened" << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl
               << "This sample is an API example, for any performance measurements "
                  "please use the dedicated benchmark_app tool"
               << slog::endl;
    return 0;
}
