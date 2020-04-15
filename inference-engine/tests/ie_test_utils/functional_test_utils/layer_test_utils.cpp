// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_test_utils.hpp"

namespace LayerTestsUtils {

FuncTestsCommon::FuncTestsCommon() {
    core = PluginCache::get().ie(targetDevice).get();
}

void FuncTestsCommon::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Configure();
    LoadNetwork();
    Infer();
    Validate();
}

FuncTestsCommon::~FuncTestsCommon() {
    if (!configuration.empty()) {
        PluginCache::get().reset();
    }
}

InferenceEngine::Blob::Ptr FuncTestsCommon::GenerateInput(const InferenceEngine::InputInfo& info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

void FuncTestsCommon::Compare(const std::vector<std::uint8_t>& expected, const InferenceEngine::Blob::Ptr& actual) {
    ASSERT_EQ(expected.size(), actual->byteSize());
    const auto& expectedBuffer = expected.data();

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const std::uint8_t*>();

    const auto& precision = actual->getTensorDesc().getPrecision();
    const auto& size = actual->size();
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            Compare(reinterpret_cast<const float*>(expectedBuffer), reinterpret_cast<const float*>(actualBuffer), size, 1e-2f);
            break;
        case InferenceEngine::Precision::I32:
            Compare(reinterpret_cast<const std::int32_t*>(expectedBuffer), reinterpret_cast<const std::int32_t*>(actualBuffer), size, 0);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void FuncTestsCommon::Configure() const {
    if (!configuration.empty()) {
        core->SetConfig(configuration, targetDevice);
    }
}

void FuncTestsCommon::LoadNetwork() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice);
    inferRequest = executableNetwork.CreateInferRequest();

    for (const auto& input : cnnNetwork.getInputsInfo()) {
        const auto& info = input.second;

        auto blob = GenerateInput(*info);
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
    }
}

void FuncTestsCommon::Infer() {
    inferRequest.Infer();
}

std::vector<InferenceEngine::Blob::Ptr> FuncTestsCommon::GetOutputs() {
    auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto& output : cnnNetwork.getOutputsInfo()) {
        const auto& name = output.first;
        outputs.push_back(inferRequest.GetBlob(name));
    }
    return outputs;
}

void FuncTestsCommon::Validate() {
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

    const auto& expectedOutputs = ngraph::helpers::interpreterFunction(function, referenceInputs);
    const auto& actualOutputs = GetOutputs();
    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
        << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];
        Compare(expected, actual);
    }
}

}  // namespace LayerTestsUtils
