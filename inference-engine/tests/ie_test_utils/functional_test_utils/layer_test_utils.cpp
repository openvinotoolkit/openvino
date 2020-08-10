// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/convert_batch_to_space.hpp>
#include <transformations/convert_space_to_batch.hpp>

#include "layer_test_utils.hpp"

namespace LayerTestsUtils {

LayerTestsCommon::LayerTestsCommon() {
    threshold = 1e-2f;
    core = PluginCache::get().ie(targetDevice);
}

LayerTestsCommon::~LayerTestsCommon() {
    if (!configuration.empty()) {
        PluginCache::get().reset();
    }
}

InferenceEngine::Blob::Ptr LayerTestsCommon::generateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

void LayerTestsCommon::configurePlugin() {
    if (!configuration.empty()) {
        core->SetConfig(configuration, targetDevice);
    }
}

void LayerTestsCommon::configureNetwork() const {
    for (const auto &in : cnnNetwork.getInputsInfo()) {
        if (inLayout != InferenceEngine::Layout::ANY) {
            in.second->setLayout(inLayout);
        }
        if (inPrc != InferenceEngine::Precision::UNSPECIFIED) {
            in.second->setPrecision(inPrc);
        }
    }

    for (const auto &out : cnnNetwork.getOutputsInfo()) {
        if (outLayout != InferenceEngine::Layout::ANY) {
            out.second->setLayout(outLayout);
        }
        if (outPrc != InferenceEngine::Precision::UNSPECIFIED) {
            out.second->setPrecision(outPrc);
        }
    }
}

void LayerTestsCommon::loadNetwork() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    configureNetwork();
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice);
}

void LayerTestsCommon::setInput() {
    configurePlugin();
    loadNetwork();
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        auto blob = generateInput(*info);
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
    }
    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
        configuration.count(InferenceEngine::PluginConfigParams::YES)) {
        auto batchSize = cnnNetwork.getInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
        inferRequest.SetBatch(batchSize);
    }
}

void LayerTestsCommon::getActualResults() {
    infer();
}

void LayerTestsCommon::infer() {
    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
        configuration.count(InferenceEngine::PluginConfigParams::YES)) {
        auto batchSize = cnnNetwork.getInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
        inferRequest.SetBatch(batchSize);
    }
    inferRequest.Infer();
}

static void SetByteVectorFromBlobVector(std::vector<std::vector<std::uint8_t>> &byteVector,
        const std::vector<InferenceEngine::Blob::Ptr> blobVector) {
    byteVector.clear();
    byteVector.resize(blobVector.size());
    for (std::size_t i = 0; i < blobVector.size(); ++i) {
        const auto& blob = blobVector[i];
        const auto& vectorSize = blob->byteSize();

        auto& innerVector = byteVector[i];
        innerVector.resize(vectorSize);

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto buffer = lockedMemory.as<const std::uint8_t*>();
        std::copy(buffer, buffer + vectorSize, innerVector.data());
    }
}

std::vector<InferenceEngine::Blob::Ptr> LayerTestsCommon::getOutputs() {
    auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto &output : cnnNetwork.getOutputsInfo()) {
        const auto &name = output.first;
        outputs.push_back(inferRequest.GetBlob(name));
    }
    return outputs;
}

void LayerTestsCommon::compare(const std::vector<std::vector<std::uint8_t>>& expectedOutputs, const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) {
    SetByteVectorFromBlobVector(actualByteOutput, actualOutputs);
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualByteOutput[outputIndex];
        ASSERT_EQ(expected.size(), actual.size());
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actualOutputs[outputIndex]);
        IE_ASSERT(memory);
        const unsigned char *expectedBuffer = expected.data();
        const unsigned char *actualBuffer = actual.data();
        const InferenceEngine::Precision precision = actualOutputs[outputIndex]->getTensorDesc().getPrecision();
        auto size = actualByteOutput.size();
        // TODO:  crop blobs before comparation instead
        // With dynamic batch, you need to size
        if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED)) {
            auto batchSize = actualOutputs[outputIndex]->getTensorDesc().getDims()[0];
            auto halfBatchSize = batchSize > 1 ? batchSize/ 2 : 1;
            size = (actualOutputs[outputIndex]->size() * halfBatchSize / batchSize);
        }
        compareValues(expectedBuffer, actualBuffer, size, precision);
    }
}

void LayerTestsCommon::validate() {
    const auto& actualOutputs = getOutputs();
    SetByteVectorFromBlobVector(byteInputData, inputs);

    // nGraph interpreter does not support f16
    // IE converts f16 to f32
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(function);
    function->validate_nodes_and_infer_types();
    ::ngraph::element::Type convertType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(actualOutputs[0]->getTensorDesc().getPrecision());
    switch (refMode) {
        case FuncTestUtils::RefMode::INTERPRETER: {
            expectedByteOutput = ngraph::helpers::interpreterFunction(function, byteInputData, convertType);
            break;
        }
        case FuncTestUtils::RefMode::CONSTANT_FOLDING: {
            const auto &foldedFunc = ngraph::helpers::foldFunction(function, byteInputData);
            expectedByteOutput = ngraph::helpers::getConstData(foldedFunc, convertType);
            break;
        }
        case FuncTestUtils::RefMode::IE: {
            // reference inference on device with other options and nGraph function has to be implemented here
            break;
        }
        case FuncTestUtils::RefMode::INTERPRETER_TRANSFORMATIONS: {
            auto cloned_function = ngraph::clone_function(*function);

            // todo: add functionality to configure the necessary transformations for each test separately
            ngraph::pass::Manager m;
            m.register_pass<ngraph::pass::ConvertSpaceToBatch>();
            m.register_pass<ngraph::pass::ConvertBatchToSpace>();
            m.run_passes(cloned_function);
            expectedByteOutput = ngraph::helpers::interpreterFunction(cloned_function, byteInputData, convertType);
            break;
        }
    }

    if (expectedByteOutput.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedByteOutput.size())
    << "nGraph interpreter has " << expectedByteOutput.size() << " outputs, while IE " << actualOutputs.size();

    compare(expectedByteOutput, actualOutputs);
}
}  // namespace LayerTestsUtils
