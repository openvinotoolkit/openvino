// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <random>
#include <fstream>
#include <format_reader_ptr.h>

#include "segmentation_matcher.hpp"

static std::vector<std::vector<size_t>> blobToImageOutputArray(InferenceEngine::TBlob<float>::Ptr output,
                                                                      size_t *pWidth, size_t *pHeight,
                                                                      size_t *pChannels) {
    std::vector<std::vector<size_t>> outArray;
    size_t W = 0, C = 0, H = 0;

    auto outputDims = output->getTensorDesc().getDims();
    if (outputDims.size() == 3) {
        C = outputDims.at(0);
        H = outputDims.at(1);
        W = outputDims.at(2);
    } else if (outputDims.size() == 4) {
        C = outputDims.at(1);
        H = outputDims.at(2);
        W = outputDims.at(3);
    } else if (outputDims.size() == 5) {
        C = outputDims.at(1);
        H = outputDims.at(3);
        W = outputDims.at(4);
    } else {
        THROW_IE_EXCEPTION << "Output blob has unsupported layout " << output->getTensorDesc().getLayout();
    }

    // Get classes
    const float *outData = output->data();
    for (unsigned h = 0; h < H; h++) {
        std::vector<size_t> row;
        for (unsigned w = 0; w < W; w++) {
            float max_value = outData[h * W + w];
            size_t index = 0;
            for (size_t c = 1; c < C; c++) {
                size_t dataIndex = c * H * W + h * W + w;
                if (outData[dataIndex] > max_value) {
                    index = c;
                    max_value = outData[dataIndex];
                }
            }
            row.push_back(index);
        }
        outArray.push_back(row);
    }

    if (pWidth != nullptr) *pWidth = W;
    if (pHeight != nullptr) *pHeight = H;
    if (pChannels != nullptr) *pChannels = C;

    return outArray;
}

namespace Regression { namespace Matchers {

void SegmentationMatcher::match() {
    // Read network
    std::string binFileName = testing::FileUtils::fileNameNoExt(config._path_to_models) + ".bin";
    auto network = config.ie_core->ReadNetwork(config._path_to_models, binFileName);

    // Change batch size if it is not equal 1
    InferenceEngine::InputsDataMap inputs;
    inputs = network.getInputsInfo();
    ASSERT_EQ(inputs.size() ,1);
    InferenceEngine::InputInfo::Ptr ii = inputs.begin()->second;

    InferenceEngine::SizeVector inputDims = ii->getTensorDesc().getDims();
    if (inputDims.at(0) != 1) {
        std::cerr << "[WARNING]: Batch size will be equal 1." << std::endl;
        network.setBatchSize(1);
        inputDims = ii->getTensorDesc().getDims();
    }

    InferenceEngine::OutputsDataMap outInfo;
    outInfo = network.getOutputsInfo();
    ASSERT_EQ(outInfo.size(), 1);
    ASSERT_NE(outInfo.begin()->second, nullptr);

    InferenceEngine::SizeVector outputDims = outInfo.begin()->second->getDims();

    if (outputDims.size() != 4) {
        THROW_IE_EXCEPTION << "Incorrect output dimensions for Deconvolution model";
    }

    // Read image
    FormatReader::ReaderPtr reader(config._paths_to_images[0].c_str());
    if (reader.get() == nullptr) {
        THROW_IE_EXCEPTION << "[ERROR]: Image " << config._paths_to_images[0] << " cannot be read!";
    }

    int inputNetworkSize = static_cast<int>(std::accumulate(
        inputDims.begin(), inputDims.end(), (size_t)1, std::multiplies<size_t>()));

    if (reader->size() != inputNetworkSize) {
        THROW_IE_EXCEPTION << "[ERROR]: Input sizes mismatch, got " << reader->size() << " bytes, expecting "
                           << inputNetworkSize;
    }

    // Allocate blobs
    InferenceEngine::Blob::Ptr input;
    switch (inputs.begin()->second->getPrecision()) {
        case InferenceEngine::Precision::FP32 :
            input = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, inputDims, NCHW });
            break;
        case InferenceEngine::Precision::Q78 :
        case InferenceEngine::Precision::I16 :
            input = InferenceEngine::make_shared_blob<short>({ InferenceEngine::Precision::I16, inputDims, NCHW });
            break;
        case InferenceEngine::Precision::U8 :
            input = InferenceEngine::make_shared_blob<uint8_t>({ InferenceEngine::Precision::U8, inputDims, NCHW });
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported network precision: " << inputs.begin()->second->getPrecision();
    }
    input->allocate();

    output = InferenceEngine::make_shared_blob<float>(outInfo.begin()->second->getTensorDesc());
    output->allocate();

    // Load image to blob
    ConvertImageToInput(reader->getData().get(), reader->size(), *input);

    InferenceEngine::ResponseDesc dsc;
    InferenceEngine::StatusCode sts;

    auto loadedExecutableNetwork = config.ie_core->LoadNetwork(network, config._device_name, config.plugin_config);
    InferenceEngine::ExecutableNetwork executableNetwork;
    if (config.useExportImport) {
        std::stringstream stream;
        loadedExecutableNetwork.Export(stream);
        executableNetwork = config.ie_core->ImportNetwork(stream);
    } else {
        executableNetwork = loadedExecutableNetwork;
    }

    auto inferRequest = executableNetwork.CreateInferRequest();
    inferRequest.SetBlob(inputs.begin()->first.c_str(), input);
    inferRequest.SetBlob(outInfo.begin()->first.c_str(), output);

    // Infer model
    inferRequest.Infer();

    // Convert output data and save it to image
    outArray = blobToImageOutputArray(output, nullptr, nullptr, &C);
}

float SegmentationMatcher::compareOutputBmp(std::vector<std::vector<size_t>> data, size_t classesNum, const std::string& inFileName) {
    unsigned int seed = (unsigned int)time(NULL);
    std::vector<Color> colors = {
        {128, 64,  128},
        {232, 35,  244},
        {70,  70,  70},
        {156, 102, 102},
        {153, 153, 190},
        {153, 153, 153},
        {30,  170, 250},
        {0,   220, 220},
        {35,  142, 107},
        {152, 251, 152},
        {180, 130, 70},
        {60,  20,  220},
        {0,   0,   255},
        {142, 0,   0},
        {70,  0,   0},
        {100, 60,  0},
        {90,  0,   0},
        {230, 0,   0},
        {32,  11,  119},
        {0,   74,  111},
        {81,  0,   81}
    };
    while (classesNum > colors.size()) {
        static std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 255);
        Color color(dist(rng), dist(rng), dist(rng));
        colors.push_back(color);
    }


    FormatReader::ReaderPtr rd(inFileName.c_str());
    if (rd.get() == nullptr) {
        THROW_IE_EXCEPTION << "[ERROR]: Image " << inFileName << " cannot be read!";
    }

    auto height = data.size();
    auto width = data.at(0).size();

    if (rd.get()->width() != width || rd.get()->height() != height) {
        return 0.0;
    }

    float rate = 0.0;

    unsigned char* pixels = rd.get()->getData().get();

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            unsigned char pixel[3];
            size_t index = data.at(y).at(x);
            pixel[0] = colors.at(index).red();
            pixel[1] = colors.at(index).green();
            pixel[2] = colors.at(index).blue();

            unsigned char pixelR[3];
            pixelR[0] = pixels[(y*width + x)*3 + 0];
            pixelR[1] = pixels[(y*width + x)*3 + 1];
            pixelR[2] = pixels[(y*width + x)*3 + 2];

            if (pixel[0] == pixelR[0] &&
                pixel[1] == pixelR[1] &&
                pixel[2] == pixelR[2]) {

                rate ++;
            }
        }
    }

    rate /= (width * height);
    return rate;
}

void SegmentationMatcher::checkResult(std::string imageFileName) {
    std::ifstream inFile;

    float rate = compareOutputBmp(outArray, C, TestDataHelpers::get_data_path() + "/test_results/" + imageFileName/*ifs*/);

    float dist = 1.0f - rate;
    if (dist > config.nearValue) {
        FAIL() << "Comparison distance " << dist << " is greater than " << config.nearValue;
    } else {
        std::cout << "Comparison distance " << dist << " is smaller than " << config.nearValue << std::endl;
    }
}

} }  //  namespace matchers
