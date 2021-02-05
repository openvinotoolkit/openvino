// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample_async/main.cpp
* @example classification_sample_async/main.cpp
*/

#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <condition_variable>
#include <mutex>

#include <inference_engine.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>

#include <sys/stat.h>

#include "classification_sample_async.h"

#include <hetero/hetero_plugin_config.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/ops.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>

using namespace InferenceEngine;

void insert_noop_reshape_after(std::shared_ptr<ngraph::Node>& node, const std::string& opName) {
    // Get all consumers for node
    const auto consumers = node->output(0).get_target_inputs();

    // Create constant with values {0, 1, 2, 3}
    auto constant = std::make_shared<ngraph::op::Constant>(
                ngraph::element::i64, ngraph::Shape{node->get_shape().size()}, std::vector<size_t>{0, 1, 2, 3});
    constant->set_friendly_name(opName + "_const");
    constant->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>("VPUX");

    // Create noop transpose node
    auto transpose = std::make_shared<ngraph::opset1::Transpose>(node, constant);
    transpose->set_friendly_name(opName);
    transpose->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>("VPUX");

    // Reconnect all consumers to the new node
    for (auto input : consumers) {
        input.replace_source_output(transpose);
    }
}

// Sample function that cuts network in half
void assignAffinities(InferenceEngine::CNNNetwork& network, const std::string& firstDevice,
                      const std::string& secondDevice, std::string splitLayer) {
    slog::info << "Manual distribution logic is used\n";

    auto ngraphFunction = network.getFunction();

    auto orderedOps = ngraphFunction->get_ordered_ops();
    auto lastSubgraphLayer =
            std::find_if(begin(orderedOps), end(orderedOps), [&](const std::shared_ptr<ngraph::Node>& node) {
                return splitLayer == node->get_friendly_name();
            });

    if (lastSubgraphLayer == end(orderedOps)) {
        slog::err << "Splitting layer \"" << splitLayer << "\" was not found.";
    }

    // with VPUX plugin, also add temporary SW layer at the end of the subnetwork
    if (firstDevice == "VPUX") {
        splitLayer = "last_reshape_layer";
        insert_noop_reshape_after(*lastSubgraphLayer, splitLayer);
        orderedOps = ngraphFunction->get_ordered_ops();
    }

    // split network into two parts using affinity
    auto deviceName = std::string{firstDevice};
    for (auto&& node : orderedOps) {
        auto& nodeInfo = node->get_rt_info();
        nodeInfo["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(deviceName);

        if (splitLayer == node->get_friendly_name()) {
            deviceName = secondDevice;
        }
    }
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

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

int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Creating Inference Engine" << slog::endl;

        const auto deviceName = "HETERO:" + FLAGS_d;

        Core ie;

        /** Printing device version **/
        slog::info << ie.GetVersions(deviceName) << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        slog::info << "Loading network files" << slog::endl;

        /** Read network model **/
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);

        const auto devices = [&]() -> std::pair<std::string, std::string> {
            const auto comma = std::find(begin(FLAGS_d), end(FLAGS_d), ',');
            auto firstDevice = std::string{begin(FLAGS_d), comma};
            auto secondDevice = std::string{next(comma), end(FLAGS_d)};
            slog::info << "Using device: HETERO:" << firstDevice << "," << secondDevice << slog::endl;
            return {std::move(firstDevice), std::move(secondDevice)};
        }();

        assignAffinities(network, devices.first, devices.second, FLAGS_split_layer);

        slog::info << "The topology " << network.getName() << " will be run on HETERO:"
                   << devices.first << "," << devices.second << " device" << slog::endl;

        // --------------------------- 3. Configure input & output ---------------------------------------------
        if (network.getOutputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 output only");

        slog::info << "Preparing input blobs" << slog::endl;

        InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");

        auto inputInfoItem = *inputInfo.begin();
        inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setLayout(Layout::NHWC);

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
                    reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
                                    inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data != nullptr) {
                imagesData.push_back(data);
                validImageNames.push_back(i);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork executable_network = ie.LoadNetwork(network, deviceName,
            {{"VPU_COMPILER_REMOVE_PERMUTE_NOOP", "NO"}});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest inferRequest = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------

        Blob::Ptr inputBlob = inferRequest.GetBlob(inputInfoItem.first);
        SizeVector dims = inputBlob->getTensorDesc().getDims();
        /** Fill input tensor with images. First b channel, then g and r channels **/
        size_t C = dims[1];
        size_t H = dims[2];
        size_t W = dims[3];

        MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
        if (!minput) {
            slog::err << "We expect MemoryBlob from inferRequest, but by fact we were not able to cast inputBlob to MemoryBlob" << slog::endl;
            return 1;
        }

        // locked memory holder should be alive all time while access to its buffer happens
        auto minputHolder = minput->wmap();
        auto data = minputHolder.as<PrecisionTrait<Precision::U8>::value_type *>();

        // Set RGB or BGR image should be passed to the network
        const bool isBGR = false;

        /** Iterate over all input images **/
        for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
            auto image = imagesData.at(image_id).get();
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    for (size_t c = 0; c < C; ++c) {
                        if (isBGR) {
                            data[image_id * H * W * C + c + w * C + h * W * C] = image[c + w * C + h * W * C];
                        } else {
                            data[image_id * H * W * C + c + w * C + h * W * C] = image[(C - c - 1) + w * C + h * W * C];
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        size_t numIterations = 10;
        size_t curIteration = 0;
        std::condition_variable condVar;

        inferRequest.SetCompletionCallback(
                [&] {
                    curIteration++;
                    slog::info << "Completed " << curIteration << " async request execution" << slog::endl;
                    if (curIteration < numIterations) {
                        /* here a user can read output containing inference results and put new input
                           to repeat async request again */
                        inferRequest.StartAsync();
                    } else {
                        /* continue sample execution after last Asynchronous inference request execution */
                        condVar.notify_one();
                    }
                });

        /* Start async request for the first time */
        slog::info << "Start inference (" << numIterations << " asynchronous executions)" << slog::endl;
        inferRequest.StartAsync();

        /* Wait all repetitions of the async request */
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&]{ return curIteration == numIterations; });

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;
        OutputsDataMap outputInfo(network.getOutputsInfo());
        if (outputInfo.empty())
            throw std::runtime_error("Can't get output blobs");
        Blob::Ptr outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);

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

        ClassificationResult classificationResult(outputBlob, validImageNames,
                                                  batchSize, 10,
                                                  labels);
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl << "This sample is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool" << slog::endl;
    return 0;
}
