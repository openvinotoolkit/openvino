// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_layers_tests.hpp"

#include <thread>
#include <chrono>
#include <iostream>

#include "functional_test_utils/plugin_cache.hpp"
#include "ie_memcpy.h"
#include "common_layers_params.hpp"

#include <vpu/utils/error.hpp>

#include "blob_factory.hpp"
#include "debug.h"
#include "vpu_tests_config.hpp"

using namespace InferenceEngine;

void vpuLayersTests::SetUp() {
    _vpuPluginPtr = std::make_shared<IECoreAdapter>(PluginCache::get().ie(), vpu::tests::deviceName());

    _genDataCallback = GenRandomData;
    TestsCommon::SetUp();
    SetSeed(DEFAULT_SEED_VALUE);
}

void vpuLayersTests::TearDown() {
    if (auto test_info = testing::UnitTest::GetInstance()->current_test_info()) {
        if (auto type_param = test_info->type_param()) {
            std::cout << "[ TYPE     ] \t" << type_param << std::endl;
        }
        if (auto value_param = test_info->value_param()) {
            std::cout << "[ VALUE    ] \t" << value_param << std::endl;
        }

        if (auto dumpModelsPath = std::getenv("IE_VPU_DUMP_LAYER_TESTS_MODELS_DIRECTORY")) {
            std::string testName = test_info->name();
            std::replace(testName.begin(), testName.end(), '/', '_');

            auto filename = dumpModelsPath + std::string("/") + testName;

            std::string xmlName = filename + ".xml";
            std::string weightsName = filename + ".bin";
            _cnnNetwork.serialize(xmlName, weightsName);

            std::string blobName = filename + ".blob";
            ASSERT_NO_THROW(_exeNetwork.Export(blobName));
        }
    }

    _exeNetwork = {};
    _inferRequest = {};
    _refBlob = {};

    TestsCommon::TearDown();
}

bool vpuLayersTests::CheckMyriadX() {
    if (auto envVar = std::getenv("IE_VPU_MYRIADX")) {
        return std::stoi(envVar) != 0;
    }
    return false;
}

void vpuLayersTests::SetSeed(uint32_t seed) {
    /*just to be able to repeat results */
    std::srand(seed);
}

Blob::Ptr vpuLayersTests::getReferenceOutput() {
    return _testNet.getLastOutput();
}

void vpuLayersTests::dumpPerformance() {
    auto perfMap = _inferRequest.GetPerformanceCounts();
    std::vector <std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
    std::sort(perfVec.begin(), perfVec.end(),
              [=](const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair1,
                  const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair2) -> bool {
                  return pair1.second.execution_index < pair2.second.execution_index;
              });

    for (auto it = perfVec.begin(); it != perfVec.end(); ++it) {
        std::string layerName = it->first;
        InferenceEngine::InferenceEngineProfileInfo info = it->second;
        if (info.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
            printf("\x1B[32m[----------]\x1B[0m Myriad time = '%s' layer with '%s' type is %f ms.\n", layerName.c_str(), info.exec_type, info.realTime_uSec / 1000.f);
        }
    }
}

bool vpuLayersTests::wasCustomLayerInferred() const {
    auto perfMap = _inferRequest.GetPerformanceCounts();
    const auto isCustomLayer = [&](const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>& info) {
        return !strcmp(info.second.exec_type, "Custom");
    };
    return std::any_of(begin(perfMap), end(perfMap), isCustomLayer);
}

namespace {

template<class TensorDescriptor>
Blob::Ptr allocateBlob(const TensorDescriptor& source, bool lockLayout) {
    const auto& descriptor = source->getTensorDesc();

    // reference functions work only with NHWC layout
    const auto& outputLayout = descriptor.getLayout();
    const auto& layout = lockLayout ? outputLayout : (outputLayout == NHWC || outputLayout == NCHW) ? NHWC : outputLayout;

    // it is required to create new TensorDesc object: #-26746
    auto blob = make_blob_with_precision(TensorDesc{descriptor.getPrecision(), descriptor.getDims(), layout});

    blob->allocate();
    return blob;
}

template<class Blob>
void configure(const Blob& blob, const InferenceEngine::Precision& precision, const vpu::LayoutPreference& layoutPreference) {
    if (precision != InferenceEngine::Precision::UNSPECIFIED) {
        // default behavior is to set FP16 precision to avoid "Convert" layer from FP32 to FP16
        // in case of network with precision other than just FP16 or FP32 (e.g. "I32" or mixed) user changes precision to "UNSPECIFIED"
        // so precision defined in IR will be used
        blob->setPrecision(precision);
    }

    blob->setLayout(vpu::deviceLayout(blob->getLayout(), layoutPreference));
}

}

void vpuLayersTests::genInputBlobs(bool lockLayout) {
    auto genDataCallback = (_genDataCallback0 != nullptr) ? _genDataCallback0 : _genDataCallback;
    for (const auto& input : _inputsInfo) {
        auto inputBlob = allocateBlob(input.second, lockLayout);

        ASSERT_NE(genDataCallback, nullptr);
        genDataCallback(inputBlob);

        ASSERT_NO_THROW(_inferRequest.SetBlob(input.first.c_str(), inputBlob));

        _inputMap[input.first] = inputBlob;
        genDataCallback = _genDataCallback;
    }
}

void vpuLayersTests::genRefBlob(bool lockLayout) {
    _refBlob = allocateBlob(_outputsInfo.begin()->second, lockLayout);
}

void vpuLayersTests::genOutputBlobs(bool lockLayout) {
    for (const auto& output : _outputsInfo) {
        auto outputBlob = allocateBlob(output.second, lockLayout);

        ASSERT_NO_THROW(_inferRequest.SetBlob(output.first.c_str(), outputBlob));

        _outputMap[output.first] = outputBlob;
    }
}

void vpuLayersTests::createInferRequest(const NetworkParams& params) {
    for (auto& input : _inputsInfo) {
        configure(input.second, params._inputPrecision, params._layoutPreference);
    }

    for (auto& output : _outputsInfo) {
        configure(output.second, params._outputPrecision, params._layoutPreference);
    }

    std::map<std::string, std::string> config(_config);
    if (params._useHWOpt) {
        config[InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION] = CONFIG_VALUE(YES);
    } else {
        config[InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION] = CONFIG_VALUE(NO);
    }
#if 0
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
#endif
    config[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
    config[InferenceEngine::MYRIAD_PERF_REPORT_MODE] = InferenceEngine::MYRIAD_PER_STAGE;

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork, config));
    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());

    genInputBlobs(params._lockLayout);
    genOutputBlobs(params._lockLayout);
    genRefBlob(params._lockLayout);
}

void vpuLayersTests::makeSingleLayerNetworkImpl(const LayerParams& layerParams,
                                                const NetworkParams& networkParams,
                                                const WeightsBlob::Ptr& weights) {
    IE_ASSERT(!layerParams._layerType.empty());

    if (_doReshape) {
        auto reshapedInput = _inputTensors;
        reshapedInput[0].insert(reshapedInput[0].begin(), 4 - _inputTensors[0].size(), 1);
        _testNet.addLayer(VpuTestNet::LayerInitParams("Reshape")
                 .params({})
                 .in({reshapedInput})
                 .out({_inputTensors}));
    }
    VpuTestNet::CalcWeights weightsCallback, biasesCallback;
    if (weights) {
        auto* weightsPtr = weights->data().as<uint16_t*>();
        auto* biasesPtr  = weightsPtr + layerParams._weightsSize;
        weightsCallback = [weightsPtr](uint16_t* ptr, size_t weightsSize){ memcpy(ptr, weightsPtr, weightsSize * sizeof (uint16_t)); };
        biasesCallback  = [biasesPtr ](uint16_t* ptr, size_t weightsSize){ memcpy(ptr, biasesPtr , weightsSize * sizeof (uint16_t)); };
    }
   _testNet.addLayer(VpuTestNet::LayerInitParams(layerParams._layerType)
             .params(layerParams._params)
             .in(_inputTensors)
             .out(_outputTensors)
             .weights(layerParams._weightsSize).fillWeights(std::move(weightsCallback))
             .biases(layerParams._biasesSize).fillBiases(std::move(biasesCallback)));

    genNetwork();

    if (networkParams._createInference)
        createInferRequest(networkParams);
}

void vpuLayersTests::readNetwork(const std::string& model, const WeightsBlob::Ptr& modelWeights) {
    _cnnNetwork = _vpuPluginPtr->ieCore()->ReadNetwork(model, modelWeights);

    ASSERT_NO_THROW(_inputsInfo = _cnnNetwork.getInputsInfo());
    ASSERT_NO_THROW(_outputsInfo = _cnnNetwork.getOutputsInfo());
}

void vpuLayersTests::readNetwork(const std::string& modelFilename, const std::string& weightsFilename) {
    _cnnNetwork = PluginCache::get().ie()->ReadNetwork(modelFilename, weightsFilename);

    ASSERT_NO_THROW(_inputsInfo = _cnnNetwork.getInputsInfo());
    ASSERT_NO_THROW(_outputsInfo = _cnnNetwork.getOutputsInfo());
}

bool vpuLayersTests::Infer() {
    if (_inputMap.empty() || _outputMap.empty())
        return false;

    _inferRequest.Infer();
//    dumpPerformance();

    if (!_config[InferenceEngine::MYRIAD_CUSTOM_LAYERS].empty()) {
        EXPECT_TRUE(wasCustomLayerInferred())
            << "CustomBindings.xml has been provided but Custom layer was not inferred";
    }
    return true;
}

bool vpuLayersTests::generateNetAndInfer(const NetworkParams& params) {
    genNetwork();
    createInferRequest(params);
    if (params._runRefGraph) {
        ReferenceGraph();
    }
    return Infer();
}

void vpuLayersTests::ResetGeneratedNet() {
    SetSeed(DEFAULT_SEED_VALUE);
    _exeNetwork = {};
    _inferRequest = {};
}

void vpuLayersTests::ResetReferenceLayers() {
    _testNet.clear();
}

void vpuLayersTests::SetInputReshape() {
    _doReshape = true;
}

void vpuLayersTests::SetInputTensor(const tensor_test_params & tensor) {
    _inputTensors = {tensor.asVector()};
}

void vpuLayersTests::SetInputTensor(const tensor_test_params_3d& tensor) {
    _inputTensors = {tensor.asVector()};
}

void vpuLayersTests::SetInputTensors(const IN_OUT_desc& in_tensors) {
    _inputTensors = in_tensors;
}

void vpuLayersTests::SetOutputTensor(const tensor_test_params& tensor) {
    _outputTensors = {tensor.asVector()};
}

void vpuLayersTests::SetOutputTensor(const tensor_test_params_3d& tensor) {
    _outputTensors = {tensor.asVector()};
}

void vpuLayersTests::SetOutputTensors(const IN_OUT_desc& out_tensors) {
    _outputTensors = out_tensors;
}

void vpuLayersTests::SetFirstInputToRange(float start, float finish) {
    ASSERT_NE(_inputMap.size(), 0);
    ASSERT_LT(start, finish);
    float range = finish - start;
    /* input data preparation */
    auto inputBlob = _inputMap[_inputsInfo.begin()->first];
    uint16_t *inputBlobRawDataFp16 = inputBlob->buffer().as<uint16_t*>();
    ASSERT_NE(inputBlobRawDataFp16, nullptr);
    /* values generation in the range (start, finish) to check difference with float output */
    size_t count = inputBlob->size();
    float shift = range / count;
    float i = start;
    for (size_t indx = 0; indx < count; i += shift, indx++) {
        inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(i);
    }
}

void vpuLayersTests::SetInputInOrder() {
    ASSERT_NE(_inputsInfo.size(), 0);
    auto inputBlob = _inputMap[_inputsInfo.begin()->first];
    ASSERT_NE(inputBlob, nullptr);
    uint16_t *inputBlobRawDataFp16 = inputBlob->buffer().as<uint16_t*>();
    ASSERT_NE(inputBlobRawDataFp16, nullptr);
    /* values generation in the range (-BOUND, BOUND) to check difference with float output */
    int  count = inputBlob->size();

    for (int indx = 0; indx < count; indx++) {
        inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16((float)indx);
    }
}

void vpuLayersTests::SetInputInOrderReverse() {
    ASSERT_NE(_inputsInfo.size(), 0);
    auto inputBlob = _inputMap[_inputsInfo.begin()->first];
    ASSERT_NE(inputBlob, nullptr);
    uint16_t *dstPtr = inputBlob->buffer().as<uint16_t*>();
    ASSERT_NE(dstPtr, nullptr);
    size_t count = inputBlob->size();
    for (size_t indx = 0; indx < count; indx++) {
        dstPtr[indx] = PrecisionUtils::f32tof16((float)(count - 1 - indx));
    }
}

void vpuLayersTests::genNetwork() {
    const auto& networkData = _testNet.genNetwork(_irVersion);
    readNetwork(networkData.model, networkData.weights);
    ASSERT_NE(_cnnNetwork.layerCount(), 0);
    ASSERT_GE(_cnnNetwork.getInputsInfo().size(), 1);
    ASSERT_GE(_cnnNetwork.getOutputsInfo().size(), 1);
}

void vpuLayersTests::ReferenceGraph() {
    /* data preparation */
    ASSERT_EQ(_inputsInfo.size(), 1);
    ASSERT_TRUE(!_testNet.empty());
    auto referenceInput = _testNet.getFirstInput();
    auto realInput = _inputMap[_inputsInfo.begin()->first];
    ASSERT_NE(referenceInput, nullptr);
    ASSERT_NE(realInput, nullptr);
    const size_t count = referenceInput->size();
    ASSERT_EQ(count, realInput->size());
    const uint16_t* inputBlobRawDataFp16 = realInput->buffer();
    uint16_t* refBlobRawDataFp16 = referenceInput->buffer();
    ASSERT_NE(inputBlobRawDataFp16, nullptr);
    ie_memcpy(refBlobRawDataFp16, realInput->byteSize(), inputBlobRawDataFp16, count * sizeof(uint16_t));
    _testNet.run();
}
