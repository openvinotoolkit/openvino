// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>

#include "layer_test_utils.hpp"

namespace LayerTestsUtils {

LayerTestsCommon::LayerTestsCommon() : threshold(1e-2f) {
    core = PluginCache::get().ie(targetDevice);
}

void LayerTestsCommon::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LoadNetwork();
    Infer();
    Validate();
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
    const auto &size = actual->size();
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

void LayerTestsCommon::Compare(const InferenceEngine::Blob::Ptr &expected, const InferenceEngine::Blob::Ptr &actual) {
    auto get_raw_buffer = [] (const InferenceEngine::Blob::Ptr &blob) {
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        return lockedMemory.as<const std::uint8_t *>();
    };
    const auto expectedBuffer = get_raw_buffer(expected);
    const auto actualBuffer = get_raw_buffer(actual);

    const auto &precision = actual->getTensorDesc().getPrecision();
    const auto &size = actual->size();
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
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
}

void LayerTestsCommon::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        auto blob = GenerateInput(*info);
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
    }
    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
        configuration.count(InferenceEngine::PluginConfigParams::YES)) {
        auto batchSize = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
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

    auto ieOutPrc = outPrc;
    if (outPrc == InferenceEngine::Precision::UNSPECIFIED) {
        const auto &actualOutputs = GetOutputs();
        ieOutPrc = actualOutputs[0]->getTensorDesc().getPrecision();
    }

    const auto &convertType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(ieOutPrc);
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
        case INTERPRETER_TRANSFORMATIONS: {
            auto cloned_function = ngraph::clone_function(*function);

            // todo: add functionality to configure the necessary transformations for each test separately
            ngraph::pass::Manager m;
            m.register_pass<ngraph::pass::ConvertSpaceToBatch>();
            m.register_pass<ngraph::pass::ConvertBatchToSpace>();
            m.run_passes(cloned_function);
            expectedOutputs = ngraph::helpers::interpreterFunction(cloned_function, referenceInputs, convertType);
            break;
        }
    }

    return expectedOutputs;
}

std::vector<InferenceEngine::Blob::Ptr> LayerTestsCommon::GetOutputs() {
    auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto &output : executableNetwork.GetOutputsInfo()) {
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
