// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_test_utils.hpp"

namespace LayerTestsUtils {

LayerTestsCommon::LayerTestsCommon() : threshold(1e-2f) {
    core = PluginCache::get().ie(targetDevice);
}

void LayerTestsCommon::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ConfigurePlugin();
    LoadNetwork();
    Infer();
    Validate();
}

LayerTestsCommon::~LayerTestsCommon() {
    if (!configuration.empty()) {
        PluginCache::get().reset();
    }
}

InferenceEngine::Blob::Ptr LayerTestsCommon::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

void LayerTestsCommon::Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual) {
    ASSERT_EQ(expected.size(), actual->byteSize());
    const auto &expectedBuffer = expected.data();

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

    const auto &precision = actual->getTensorDesc().getPrecision();
//    auto halfBatchSize = 1;
//    auto batchSize = 1;
    auto bufferSize = actual->size();
    // With dynamic batch, you need to size
    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED)) {
        auto batchSize = actual->getTensorDesc().getDims()[0];
        auto halfBatchSize = batchSize > 1 ? batchSize/ 2 : 1;
        bufferSize = (actual->size() * halfBatchSize / batchSize);
    }
    const auto &size = bufferSize;
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            Compare(reinterpret_cast<const float *>(expectedBuffer), reinterpret_cast<const float *>(actualBuffer),
                    size, threshold);
            break;
        case InferenceEngine::Precision::I32:
            Compare(reinterpret_cast<const std::int32_t *>(expectedBuffer),
                    reinterpret_cast<const std::int32_t *>(actualBuffer), size, 0);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void LayerTestsCommon::ConfigurePlugin() {
    if (!configuration.empty()) {
        core->SetConfig(configuration, targetDevice);
    }
}

void LayerTestsCommon::ConfigureNetwork() const {
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

void LayerTestsCommon::LoadNetwork() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    ConfigureNetwork();
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice);
}

void LayerTestsCommon::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        auto blob = GenerateInput(*info);
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
    }
    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
        configuration.count(InferenceEngine::PluginConfigParams::YES)) {
        auto batchSize = cnnNetwork.getInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
        inferRequest.SetBatch(batchSize);
    }
    inferRequest.Infer();
}

std::vector<std::vector<std::uint8_t>> LayerTestsCommon::CalculateRefs() {
    // nGraph interpreter does not support f16
    // IE converts f16 to f32
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(function);
    function->validate_nodes_and_infer_types();
    auto referenceInputs = std::vector<std::vector<std::uint8_t>>(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        const auto& inputSize = input->byteSize();

        auto& referenceInput = referenceInputs[i];
        referenceInput.resize(inputSize);

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto buffer = lockedMemory.as<const std::uint8_t*>();
        std::copy(buffer, buffer + inputSize, referenceInput.data());
    }

    const auto &actualOutputs = GetOutputs();
    const auto &convertType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(actualOutputs[0]->getTensorDesc().getPrecision());
    std::vector<std::vector<std::uint8_t>> expectedOutputs;
    switch (refMode) {
        case INTERPRETER: {
            expectedOutputs = ngraph::helpers::interpreterFunction(function, referenceInputs, convertType);
            break;
        }
        case CONSTANT_FOLDING: {
            const auto &foldedFunc = ngraph::helpers::foldFunction(function, referenceInputs);
            expectedOutputs = ngraph::helpers::getConstData(foldedFunc, convertType);
            break;
        }
        case IE: {
            // reference inference on device with other options and nGraph function has to be implemented here
            break;
        }
    }

    return expectedOutputs;
}

std::vector<InferenceEngine::Blob::Ptr> LayerTestsCommon::GetOutputs() {
    auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto &output : cnnNetwork.getOutputsInfo()) {
        const auto &name = output.first;
        outputs.push_back(inferRequest.GetBlob(name));
    }
    return outputs;
}

void LayerTestsCommon::Compare(const std::vector<std::vector<std::uint8_t>>& expectedOutputs, const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) {
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];
        Compare(expected, actual);
    }
}

void LayerTestsCommon::Validate() {
    auto expectedOutputs = CalculateRefs();
    const auto& actualOutputs = GetOutputs();

    if (expectedOutputs.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
        << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    Compare(expectedOutputs, actualOutputs);
}

void LayerTestsCommon::SetRefMode(RefMode mode) {
    refMode = mode;
}
}  // namespace LayerTestsUtils
