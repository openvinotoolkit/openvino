// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point the Inference Engine sample application
 * @file classification_sample_async/main.cpp
 * @example classification_sample_async/main.cpp
 */

#include <format_reader_ptr.h>
#include <samples/classification_results.h>
#include <sys/stat.h>

#include <condition_variable>
#include <fstream>
#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <vector>

#include "classification_sample_async.h"

using namespace InferenceEngine;

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_nt <= 0) {
        throw std::logic_error("Incorrect value for nt argument. It should be greater than 0.");
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
        // ------------------------------ Read input
        // -----------------------------------------------------------
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
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty() && (FLAGS_d == "GPU" || FLAGS_d == "MYRIAD" || FLAGS_d == "HDDL")) {
            // Config for device plugin custom extension is loaded from an .xml
            // description
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, FLAGS_d);
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
        if (network.getOutputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 output only");

        // --------------------------- Prepare input blobs
        // -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1)
            throw std::logic_error("Sample supports topologies with 1 input only");

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the device **/
        inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::shared_ptr<unsigned char>> imagesData = {};
        std::vector<std::string> validImageNames = {};
        for (const auto& i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(
                reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data != nullptr) {
                imagesData.push_back(data);
                validImageNames.push_back(i);
            }
        }
        if (imagesData.empty() || validImageNames.empty())
            throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Loading model to the device
        // ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create infer request
        // -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest inferRequest = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------
        for (auto& item : inputInfo) {
            Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);
            SizeVector dims = inputBlob->getTensorDesc().getDims();
            /** Fill input tensor with images. First b channel, then g and r channels
             * **/
            size_t num_channels = dims[1];
            size_t image_size = dims[3] * dims[2];

            MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
            if (!minput) {
                slog::err << "We expect MemoryBlob from inferRequest, but by fact we "
                             "were not able to cast inputBlob to MemoryBlob"
                          << slog::endl;
                return 1;
            }
            // locked memory holder should be alive all time while access to its
            // buffer happens
            auto minputHolder = minput->wmap();

            auto data = minputHolder.as<PrecisionTrait<Precision::U8>::value_type*>();
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
                        data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid * num_channels + ch];
                    }
                }
            }
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 7. Do inference
        // ---------------------------------------------------------
        size_t numIterations = 10;
        size_t curIteration = 0;
        std::condition_variable condVar;

        inferRequest.SetCompletionCallback([&] {
            curIteration++;
            slog::info << "Completed " << curIteration << " async request execution" << slog::endl;
            if (curIteration < numIterations) {
                /* here a user can read output containing inference results and put new
                   input to repeat async request again */
                inferRequest.StartAsync();
            } else {
                /* continue sample execution after last Asynchronous inference request
                 * execution */
                condVar.notify_one();
            }
        });

        /* Start async request for the first time */
        slog::info << "Start inference (" << numIterations << " asynchronous executions)" << slog::endl;
        inferRequest.StartAsync();

        /* Wait all repetitions of the async request */
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&] {
            return curIteration == numIterations;
        });

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 8. Process output
        // -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;
        OutputsDataMap outputInfo(network.getOutputsInfo());
        if (outputInfo.empty())
            throw std::runtime_error("Can't get output blobs");
        Blob::Ptr outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);

        /** Validating -nt value **/
        const size_t resultsCnt = outputBlob->size() / batchSize;
        if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
            slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " << resultsCnt + 1
                       << " and more than 0)\n            Maximal value " << resultsCnt << " will be used." << slog::endl;
            FLAGS_nt = resultsCnt;
        }

        /** Read labels from file (e.x. AlexNet.labels) **/
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
        ClassificationResult classificationResult(outputBlob, validImageNames, batchSize, FLAGS_nt, labels);
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl
               << "This sample is an API example, for any performance measurements "
                  "please use the dedicated benchmark_app tool"
               << slog::endl;
    return 0;
}
