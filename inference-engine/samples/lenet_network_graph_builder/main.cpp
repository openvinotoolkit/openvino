// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <vector>
#include <string>
#include <memory>

#include <inference_engine.hpp>
#include <ie_builders.hpp>
#include <ie_utils.hpp>
#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include <gflags/gflags.h>
#include "lenet_network_graph_builder.hpp"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_ni <= 0) {
        throw std::logic_error("Incorrect value for ni argument. It should be more than 0");
    }

    if (FLAGS_nt <= 0 || FLAGS_nt > 10) {
        throw std::logic_error("Incorrect value for nt argument. It should be more than 0 and less than 10");
    }

    return true;
}

void readFile(const std::string &file_name, void *buffer, size_t maxSize) {
    std::ifstream inputFile;

    inputFile.open(file_name, std::ios::binary | std::ios::in);
    if (!inputFile.is_open()) {
        throw std::logic_error("cannot open file weight file");
    }
    if (!inputFile.read(reinterpret_cast<char *>(buffer), maxSize)) {
        inputFile.close();
        throw std::logic_error("cannot read bytes from weight file");
    }

    inputFile.close();
}

TBlob<uint8_t>::CPtr ReadWeights(std::string filepath) {
    std::ifstream weightFile(filepath, std::ifstream::ate | std::ifstream::binary);
    int64_t fileSize = weightFile.tellg();

    if (fileSize < 0) {
        throw std::logic_error("Incorrect weight file");
    }

    size_t ulFileSize = static_cast<size_t>(fileSize);

    TBlob<uint8_t>::Ptr weightsPtr(new TBlob<uint8_t>(Precision::FP32, C, {ulFileSize}));
    weightsPtr->allocate();
    readFile(filepath, weightsPtr->buffer(), ulFileSize);

    return weightsPtr;
}

/**
 * @brief The entry point for inference engine automatic squeezenet networt builder sample
 * @file squeezenet networt builder/main.cpp
 * @example squeezenet networt builder/main.cpp
 */
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        if (images.empty()) {
            throw std::logic_error("No suitable images were found");
        }

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""}).getPluginByDevice(FLAGS_d);
        printPluginVersion(plugin, std::cout);

        /** Per layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        //--------------------------- 2. Create network using graph builder ------------------------------------
        TBlob<uint8_t>::CPtr weightsPtr = ReadWeights(FLAGS_m);

        Builder::Network builder("LeNet");
        size_t layerId = builder.addLayer(Builder::InputLayer("data").setPort(Port({1, 1, 28, 28})));
        auto ptrWeights = make_shared_blob(TensorDesc(Precision::FP32, {500}, Layout::C),
                weightsPtr->cbuffer().as<float *>());
        auto ptrBiases = make_shared_blob(TensorDesc(Precision::FP32, {20}, Layout::C),
                weightsPtr->cbuffer().as<float *>() + 500);
        layerId = builder.addLayer({{layerId}}, Builder::ConvolutionLayer("conv1").setKernel({5, 5}).setDilation({1, 1})
                  .setGroup(1).setStrides({1, 1}).setOutDepth(20).setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0})
                  .setWeights(ptrWeights).setBiases(ptrBiases));
        layerId = builder.addLayer({{layerId}}, Builder::PoolingLayer("pool1").setExcludePad(true).setKernel({2, 2})
                  .setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0})
                  .setPoolingType(Builder::PoolingLayer::PoolingType::MAX)
                  .setRoundingType(Builder::PoolingLayer::RoundingType::CEIL).setStrides({2, 2}));
        ptrWeights = make_shared_blob(TensorDesc(Precision::FP32, {25000}, Layout::C),
                weightsPtr->cbuffer().as<float *>() + 520);
        ptrBiases = make_shared_blob(TensorDesc(Precision::FP32, {50}, Layout::C),
                weightsPtr->cbuffer().as<float *>() + 25520);
        layerId = builder.addLayer({{layerId}}, Builder::ConvolutionLayer("conv2").setDilation({1, 1}).setGroup(1)
                  .setKernel({5, 5}).setOutDepth(50).setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0})
                  .setStrides({1, 1}).setWeights(ptrWeights).setBiases(ptrBiases));
        layerId = builder.addLayer({{layerId}}, Builder::PoolingLayer("pool2").setExcludePad(true).setKernel({2, 2})
                  .setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0}).setPoolingType(Builder::PoolingLayer::PoolingType::MAX)
                  .setRoundingType(Builder::PoolingLayer::RoundingType::CEIL).setStrides({2, 2}));
        ptrWeights = make_shared_blob(TensorDesc(Precision::FP32, {400000}, Layout::C),
                weightsPtr->cbuffer().as<float *>() + 102280 / 4);
        ptrBiases = make_shared_blob(TensorDesc(Precision::FP32, {500}, Layout::C),
                weightsPtr->cbuffer().as<float *>() + 1702280 / 4);
        layerId = builder.addLayer({{layerId}}, Builder::FullyConnectedLayer("ip1").setOutputNum(500)
                  .setWeights(ptrWeights).setBiases(ptrBiases));
        layerId = builder.addLayer({{layerId}}, Builder::ReLULayer("relu1").setNegativeSlope(0.0f));
        ptrWeights = make_shared_blob(TensorDesc(Precision::FP32, {5000}, Layout::C),
                weightsPtr->cbuffer().as<float *>() + 1704280 / 4);
        ptrBiases = make_shared_blob(TensorDesc(Precision::FP32, {10}, Layout::C),
                weightsPtr->cbuffer().as<float *>() + 1724280 / 4);
        layerId = builder.addLayer({{layerId}}, Builder::FullyConnectedLayer("ip2").setOutputNum(10)
                  .setWeights(ptrWeights).setBiases(ptrBiases));
        layerId = builder.addLayer({{layerId}}, Builder::SoftMaxLayer("prob").setAxis(1));
        size_t outputId = builder.addLayer({PortInfo(layerId)}, Builder::OutputLayer("sf_out"));

        CNNNetwork network{Builder::convertToICNNNetwork(builder.build())};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        InputsDataMap inputInfo = network.getInputsInfo();
        if (inputInfo.size() != 1) {
            throw std::logic_error("Sample supports topologies only with 1 input");
        }

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the plugin **/
        inputInfoItem.second->setPrecision(Precision::FP32);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::shared_ptr<unsigned char>> imagesData;
        for (auto & i : images) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(
                    reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
                                    inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }

        if (imagesData.empty()) {
            throw std::logic_error("Valid input images were not found!");
        }

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // --------------------------- Prepare output blobs -----------------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(network.getOutputsInfo());
        std::string firstOutputName;

        for (auto & item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }

        if (outputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks having only one output");
        }

        DataPtr& output = outputInfo.begin()->second;
        auto outputName = outputInfo.begin()->first;

        const SizeVector outputDims = output->getTensorDesc().getDims();
        const int classCount = outputDims[1];

        if (classCount > 10) {
            throw std::logic_error("Incorrect number of output classes for LeNet network");
        }

        if (outputDims.size() != 2) {
            throw std::logic_error("Incorrect output dimensions for LeNet");
        }
        output->setPrecision(Precision::FP32);
        output->setLayout(Layout::NC);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;
        ExecutableNetwork exeNetwork = plugin.LoadNetwork(network, {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = exeNetwork.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        for (const auto & item : inputInfo) {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);

            /** Filling input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = input->getTensorDesc().getDims()[1];
            size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];

            auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid ] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }
        }
        inputInfo = {};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

        double total = 0.0;
        /** Start inference & calc performance **/
        for (int iter = 0; iter < FLAGS_ni; ++iter) {
            auto t0 = Time::now();
            infer_request.Infer();
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        const Blob::Ptr outputBlob = infer_request.GetBlob(firstOutputName);
        auto outputData = outputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

        /** Validating -nt value **/
        const int resultsCnt = outputBlob->size() / batchSize;
        if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
            slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " \
                      << resultsCnt+1 << " and more than 0)\n            will be used maximal value : " << resultsCnt;
            FLAGS_nt = resultsCnt;
        }

        /** This vector stores id's of top N results **/
        std::vector<unsigned> results;
        TopResults(FLAGS_nt, *outputBlob, results);

        std::cout << std::endl << "Top " << FLAGS_nt << " results:" << std::endl << std::endl;

        /** Print the result iterating over each batch **/
        for (int image_id = 0; image_id < batchSize; ++image_id) {
            std::cout << "Image " << images[image_id] << std::endl << std::endl;
            for (size_t id = image_id * FLAGS_nt, cnt = 0; cnt < FLAGS_nt; ++cnt, ++id) {
                std::cout.precision(7);
                /** Getting probability for resulting class **/
                const auto result = outputData[results[id] + image_id*(outputBlob->size() / batchSize)];
                std::cout << std::left << std::fixed << "Number: " << results[id] << "; Probability: " << result << std::endl;
            }
            std::cout << std::endl;
        }
        // -----------------------------------------------------------------------------------------------------
        std::cout << std::endl << "total inference time: " << total << std::endl;
        std::cout << "Average running time of one iteration: " << total / static_cast<double>(FLAGS_ni) << " ms" << std::endl;
        std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total << " FPS" << std::endl;
        std::cout << std::endl;
        // -----------------------------------------------------------------------------------------------------

        /** Show performance results **/
        if (FLAGS_pc) {
            printPerformanceCounts(infer_request, std::cout);
        }
    } catch  (const std::exception &ex) {
        slog::err << ex.what() << slog::endl;
        return 3;
    }
    return 0;
}