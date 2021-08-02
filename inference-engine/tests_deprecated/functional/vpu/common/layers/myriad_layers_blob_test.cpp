// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/precision_utils.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include "myriad_layers_tests.hpp"
#include "vpu_tests_config.hpp"
#include <fstream>

using namespace InferenceEngine;
using namespace ::testing;

typedef myriadLayerTestBaseWithParam<std::string> myriadBlobTests_smoke;

std::vector<char> readBinFile(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    file.unsetf(std::ios::skipws);

    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> vec;
    vec.reserve(fileSize);

    vec.insert(vec.begin(),
        std::istream_iterator<char>(file),
        std::istream_iterator<char>());

    return vec;
}

TEST_P(myriadBlobTests_smoke, CanGetSameBlobsOnSameIR) {
    std::string HWConfigValue = GetParam();

    auto fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

    const size_t countBlobsToDump = 3;
    std::vector<std::string> filenames(countBlobsToDump);
    for (int i = 0; i < countBlobsToDump; i++) {
        ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork,
            { {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, HWConfigValue } }));
        std::stringstream modelFilenameStream;
        modelFilenameStream << "spltConvConcat" << i << ".blob";
        filenames[i] = modelFilenameStream.str();
        ASSERT_NO_THROW(_exeNetwork.Export(modelFilenameStream.str()));
    }

    for (int i = 0; i < filenames.size() - 1; i++) {
        std::vector<char>  blob1 = readBinFile(filenames[i]);
        std::vector<char>  blob2 = readBinFile(filenames[i + 1]);
        ASSERT_TRUE(blob1 == blob2);
    }

    for (int i = 0; i < filenames.size(); i++) {
        std::remove(filenames[i].c_str());
    }
}

INSTANTIATE_TEST_SUITE_P(accuracy, myriadBlobTests_smoke,
    ::testing::Values(CONFIG_VALUE(YES), CONFIG_VALUE(NO))
);

using myriadBlobExportTests_smoke = myriadLayersTests_nightly;


TEST_F(myriadBlobExportTests_smoke, CanNotDoImportOnNonExistFile)
{
    ASSERT_THROW(_vpuPluginPtr->ImportNetwork("I_dont_exist.blob"),
        InferenceEngine::NetworkNotRead);
}

TEST_F(myriadBlobExportTests_smoke, CanInferImportedNetworkOnExportedBlob)
{
    auto fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork));
    std::stringstream modelFilenameStream;
    modelFilenameStream << "SplitConvConcat" << ".blob";
    ASSERT_NO_THROW(_exeNetwork.Export(modelFilenameStream.str()));

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = _vpuPluginPtr->ImportNetwork(modelFilenameStream.str()));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
    ASSERT_NO_THROW(inferRequest.Infer());
}

TEST_F(myriadBlobExportTests_smoke, CanGetPerfCountsImportedNetwork)
{
    auto fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork));
    std::stringstream modelFilenameStream;
    modelFilenameStream << "splitConvConcat" << ".blob";
    ASSERT_NO_THROW(_exeNetwork.Export(modelFilenameStream.str()));

    std::map<std::string, std::string> config = { {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)} };
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = _vpuPluginPtr->ImportNetwork(modelFilenameStream.str(), config));
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest =  importedNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
    std::map<std::string, InferenceEngineProfileInfo> perfCounts;
    ASSERT_NO_THROW(perfCounts = inferRequest.GetPerformanceCounts());

    ASSERT_NE(0, perfCounts.size());
    for (const auto &perfInfoElem : perfCounts) {
        InferenceEngineProfileInfo perfInfo = perfInfoElem.second;
        ASSERT_EQ(perfInfo.status, InferenceEngineProfileInfo::LayerStatus::EXECUTED);
        ASSERT_STREQ(perfInfo.exec_type, "UNKNOWN");
        ASSERT_STREQ(perfInfo.layer_type, "UNKNOWN");
        ASSERT_NE(perfInfo.realTime_uSec, 0);
    }
}

class myriadConfigsWithBlobImportTests_smoke: public myriadLayersTests_nightly {
protected:
    // use this stream to redirect cout to it,
    // needs to be able check output on warnings
    std::stringstream redirectCoutStream;

    void SetUp() override {
        myriadLayersTests_nightly::SetUp();
        backup = std::cout.rdbuf();
        std::cout.rdbuf(redirectCoutStream.rdbuf());
    }

    void TearDown() override {
        myriadLayersTests_nightly::TearDown();
        std::cout.rdbuf(backup);
        std::cout << redirectCoutStream.str();
    }

private:
    std::streambuf *backup;
};


TEST_F(myriadConfigsWithBlobImportTests_smoke, TryingToSetCompileOptionPrintsWarning)
{
    auto fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork));
    std::stringstream modelFilenameStream;
    modelFilenameStream << "splitConvConcat" << ".blob";
    ASSERT_NO_THROW(_exeNetwork.Export(modelFilenameStream.str()));


    std::map<std::string, std::string> config = { {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, CONFIG_VALUE(YES)},
                                                  {InferenceEngine::MYRIAD_IGNORE_UNKNOWN_LAYERS, CONFIG_VALUE(YES)},
                                                  {InferenceEngine::MYRIAD_NONE_LAYERS, CONFIG_VALUE(YES)},
                                                  {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)},
                                                  {InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, std::to_string(10)},
                                                  {InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, std::to_string(10)} };

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = _vpuPluginPtr->ImportNetwork(modelFilenameStream.str(), config));

    std::string content = redirectCoutStream.str();
    for (auto &&elem : config) {
        // TODO: remove once all options are migrated
        std::stringstream deprecatedExpectedMsgStream;
        deprecatedExpectedMsgStream << "[Warning][VPU][Config] " << elem.first;
        const auto& deprecatedMsg = deprecatedExpectedMsgStream.str();

        std::stringstream expectedMsgStream;
        expectedMsgStream << "[Warning][VPU][Configuration] Configuration option \"" << elem.first;
        const auto& msg = expectedMsgStream.str();
        ASSERT_TRUE(content.find(msg) != std::string::npos || content.find(deprecatedMsg) != std::string::npos) << msg;
    }
}

TEST_F(myriadConfigsWithBlobImportTests_smoke, TryingToSetRuntimeOptionDoesNotPrintWarning)
{
    auto fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork));
    std::stringstream modelFilenameStream;
    modelFilenameStream << "splitConvConcat" << ".blob";
    ASSERT_NO_THROW(_exeNetwork.Export(modelFilenameStream.str()));

    std::map<std::string, std::string> config = { {CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)},
                                                  {CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)},
                                                  {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)},
                                                  {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)} };
    if (vpu::tests::deviceForceReset()) {
        config.insert({InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO)});
        config.insert({VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)});
    }

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = _vpuPluginPtr->ImportNetwork(modelFilenameStream.str(), config));

    std::string content = redirectCoutStream.str();
    for (auto &&elem : config) {
        std::stringstream expectedMsgStream;
        expectedMsgStream << "Warning:" << elem.first;
        std::string msg = expectedMsgStream.str();
        ASSERT_EQ(content.find(msg), std::string::npos);
    }
}


using myriadBlobExportAccuracyDifferentCountInAndOutTests_smoke = myriadLayerTestBaseWithParam<std::vector<size_t>>;

TEST_F(myriadBlobExportAccuracyDifferentCountInAndOutTests_smoke, IsResultOfImportedAndGeneratedModelSame)
{
    SetSeed(DEFAULT_SEED_VALUE);

    auto fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

    InferenceEngine::ExecutableNetwork originalExeNetwork;
    ASSERT_NO_THROW(originalExeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork));

    ConstInputsDataMap originalInputsInfo;
    ASSERT_NO_THROW(originalInputsInfo = originalExeNetwork.GetInputsInfo());

    InferenceEngine::InferRequest orignalInferRequest;
    ASSERT_NO_THROW(orignalInferRequest = originalExeNetwork.CreateInferRequest());

    std::vector<Blob::Ptr> inputBlobs(originalInputsInfo.size());
    auto inputBlobsIt = inputBlobs.begin();
    for (const auto &inputInfo : originalInputsInfo) {
        ASSERT_NO_THROW(*inputBlobsIt = orignalInferRequest.GetBlob(inputInfo.first.c_str()));
        GenRandomData(*inputBlobsIt);
        inputBlobsIt++;
    }

    ASSERT_NO_THROW(orignalInferRequest.Infer());

    ConstOutputsDataMap orignalOutputsInfo;
    ASSERT_NO_THROW(orignalOutputsInfo = originalExeNetwork.GetOutputsInfo());

    std::vector<Blob::Ptr> originalOutputBlobs(orignalOutputsInfo.size());
    auto outputBlobsIt = originalOutputBlobs.begin();
    for (const auto &outputInfo: orignalOutputsInfo) {
        ASSERT_NO_THROW(*outputBlobsIt = orignalInferRequest.GetBlob(outputInfo.first.c_str()));
        outputBlobsIt++;
    }

    std::stringstream modelFilenameStream;
    modelFilenameStream << "exportedModel" << ".blob";
    ASSERT_NO_THROW(originalExeNetwork.Export(modelFilenameStream.str()));

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = _vpuPluginPtr->ImportNetwork(modelFilenameStream.str()));
    InferenceEngine::InferRequest importedInferRequest;
    ASSERT_NO_THROW(importedInferRequest =  importedNetwork.CreateInferRequest());

    ConstInputsDataMap importedInputsInfo;
    ASSERT_NO_THROW(importedInputsInfo = importedNetwork.GetInputsInfo());

    inputBlobsIt = inputBlobs.begin();
    for (const auto &inputInfo : importedInputsInfo) {
        ASSERT_NO_THROW(importedInferRequest.SetBlob(inputInfo.first.c_str(), *inputBlobsIt));
        inputBlobsIt++;
    }

    ASSERT_NO_THROW(importedInferRequest.Infer());

    ConstOutputsDataMap importedOutputsInfo;
    ASSERT_NO_THROW(importedOutputsInfo = importedNetwork.GetOutputsInfo());

    outputBlobsIt = originalOutputBlobs.begin();
    for (const auto &outputInfo : importedOutputsInfo) {
        Blob::Ptr importedOutputBlobPtr;
        ASSERT_NO_THROW(importedOutputBlobPtr = importedInferRequest.GetBlob(outputInfo.first.c_str()));

        CompareCommonAbsolute(importedOutputBlobPtr, *outputBlobsIt, 0.f);
        outputBlobsIt++;
    }
}


using myriadBlobExportAccuracyDifferentPrecisionOfInAndOutTests_smoke = myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::Precision, InferenceEngine::Precision>>;

TEST_P(myriadBlobExportAccuracyDifferentPrecisionOfInAndOutTests_smoke, IsResultOfImportedAndGeneratedModelSame)
{
    SetSeed(DEFAULT_SEED_VALUE);
    InferenceEngine::Precision inputPrecision = std::get<0>(GetParam());
    InferenceEngine::Precision outputPrecision = std::get<1>(GetParam());
    std::vector<size_t> inputShape;

    auto fnPtr = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

    const auto& network = _cnnNetwork;
    InputsDataMap inputsInfo = network.getInputsInfo();
    ASSERT_EQ(inputsInfo.size(), 1);
    auto inputInfo = inputsInfo.begin();
    ASSERT_NO_THROW(inputInfo->second->setPrecision(inputPrecision));

    OutputsDataMap outputsInfo = network.getOutputsInfo();
    ASSERT_EQ(outputsInfo.size(), 1);
    auto outputInfo = outputsInfo.begin();
    ASSERT_NO_THROW(outputInfo->second->setPrecision(outputPrecision));

    InferenceEngine::ExecutableNetwork originalExeNetwork;
    ASSERT_NO_THROW(originalExeNetwork = _vpuPluginPtr->LoadNetwork(network));

    InferenceEngine::InferRequest orignalInferRequest;
    ASSERT_NO_THROW(orignalInferRequest = originalExeNetwork.CreateInferRequest());

    Blob::Ptr inputBlobPtr;
    ASSERT_NO_THROW(inputBlobPtr = orignalInferRequest.GetBlob(inputInfo->first.c_str()));
    GenRandomData(inputBlobPtr);

    ASSERT_NO_THROW(orignalInferRequest.Infer());

    Blob::Ptr outputBlobPtr;
    ASSERT_NO_THROW(outputBlobPtr = orignalInferRequest.GetBlob(outputInfo->first.c_str()));

    std::stringstream modelFilenameStream;
    modelFilenameStream << "exportedModel" << ".blob";
    ASSERT_NO_THROW(originalExeNetwork.Export(modelFilenameStream.str()));

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = _vpuPluginPtr->ImportNetwork(modelFilenameStream.str()));
    InferenceEngine::InferRequest importedInferRequest;
    ASSERT_NO_THROW(importedInferRequest =  importedNetwork.CreateInferRequest());

    ConstInputsDataMap importedInputsInfo;
    ASSERT_NO_THROW(importedInputsInfo = importedNetwork.GetInputsInfo());
    ASSERT_EQ(importedInputsInfo.size(), 1);
    auto importedInputInfo = importedInputsInfo.begin();

    ASSERT_NO_THROW(importedInferRequest.SetBlob(importedInputInfo->first.c_str(), inputBlobPtr));

    ASSERT_NO_THROW(importedInferRequest.Infer());

    ConstOutputsDataMap importedOutputsInfo;
    ASSERT_NO_THROW(importedOutputsInfo = importedNetwork.GetOutputsInfo());
    ASSERT_EQ(importedOutputsInfo.size(), 1);
    auto importedOutputInfo = importedOutputsInfo.begin();

    Blob::Ptr importedOutputBlobPtr;
    ASSERT_NO_THROW(importedOutputBlobPtr = importedInferRequest.GetBlob(importedOutputInfo->first.c_str()));

    CompareCommonAbsolute(importedOutputBlobPtr, outputBlobPtr, 0.f);
}

using myriadExtraTests_smoke = myriadLayersTests_nightly;

TEST_F(myriadExtraTests_smoke, ThereIsNoSegfaultOnZeroConvolutionWeights) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    tensor_test_params input_dims = { 1, 3, 25, 25 };
    param_size kernel = { 3, 3 };
    param_size stride = { 1, 1 };
    param_size pad = { 1, 1 };
    size_t out_channels = 3;
    size_t group = 1;
    param_size dilation_factor = { 1, 1 };

    size_t out_w = (input_dims.w + 2 * pad.x - dilation_factor.x * (kernel.x - 1) - 1 + stride.x) / stride.x;
    size_t out_h = (input_dims.h + 2 * pad.y - dilation_factor.y * (kernel.y - 1) - 1 + stride.y) / stride.y;

    tensor_test_params output_dims = { 1, out_channels, out_h, out_w };

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    size_t num_weights = kernel.x * kernel.y * (input_dims.c / group) * output_dims.c;
    size_t num_bias = output_dims.c;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr =
        InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights + num_bias));
    ie_fp16* weights = weights_ptr->data().as<ie_fp16*>();
    // set a small number in FP16
    for (size_t i = 0; i < num_weights + num_bias; i++) {
        weights[i] = 0;
    }

    std::map<std::string, std::string> layer_params = {
        { "kernel-x", std::to_string(kernel.x) }
        ,{ "kernel-y", std::to_string(kernel.y) }
        ,{ "stride-x", std::to_string(stride.x) }
        ,{ "stride-y", std::to_string(stride.y) }
        ,{ "pad-x", std::to_string(pad.x) }
        ,{ "pad-y", std::to_string(pad.y) }
        ,{ "output", std::to_string(out_channels) }
        ,{ "group", std::to_string(group) }
        ,{ "dilation-x", std::to_string(dilation_factor.x) }
        ,{ "dilation-y", std::to_string(dilation_factor.y) }
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Convolution")
                                        .params(layer_params)
                                        .weights(num_weights)
                                        .biases(num_bias),
                                        NetworkInitParams().useHWOpt(true),
                                        weights_ptr));
}

static const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::U8, InferenceEngine::Precision::FP16,
                                                                        InferenceEngine::Precision::FP32};

static const std::vector<InferenceEngine::Precision> outputPrecisions = {InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32};


INSTANTIATE_TEST_SUITE_P(accuracy, myriadBlobExportAccuracyDifferentPrecisionOfInAndOutTests_smoke,
                        ::testing::Combine(::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(outputPrecisions)));
