// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "base_reference_test.hpp"

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"

using namespace InferenceEngine;

namespace reference_tests {

CommonReferenceTest::CommonReferenceTest(): targetDevice("TEMPLATE") {
    core = PluginCache::get().ie(targetDevice);
}

void CommonReferenceTest::Exec() {
    LoadNetwork();
    FillInputs();
    Infer();
    Validate();
}

void CommonReferenceTest::LoadNetwork() {
    InferenceEngine::CNNNetwork cnnNetwork(function);
    auto inputInfo = cnnNetwork.getInputsInfo();
    auto outputInfo = cnnNetwork.getOutputsInfo();
    for (const auto& param : function->get_parameters()) {
        inputInfo[param->get_friendly_name()]->setPrecision(InferenceEngine::details::convertPrecision(param->get_element_type()));
    }
    for (const auto& result : function->get_results()) {
        outputInfo[ngraph::op::util::create_ie_output_name(result->input_value(0))]->setPrecision(
            InferenceEngine::details::convertPrecision(result->get_element_type()));
    }
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice);
}

void CommonReferenceTest::FillInputs() {
    const auto& inputInfo = executableNetwork.GetInputsInfo();
    const auto& params = function->get_parameters();
    ASSERT_EQ(params.size(), inputData.size());
    ASSERT_EQ(inputInfo.size(), inputData.size());

    for (size_t i = 0; i < params.size(); i++) {
        const auto& param = params[i];
        const auto infoIt = inputInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputInfo.cend());

        const auto& info = infoIt->second;
        auto blob = make_blob_with_precision(info->getTensorDesc());
        blob->allocate();

        ASSERT_EQ(blob->byteSize(), inputData[i]->byteSize());

        MemoryBlob::Ptr mInputData = as<MemoryBlob>(inputData[i]);
        ASSERT_NE(mInputData, nullptr);
        auto minputDataHolder = mInputData->rmap();

        MemoryBlob::Ptr mBlob = as<MemoryBlob>(blob);
        ASSERT_NE(mBlob, nullptr);
        auto mBlobHolder = mBlob->wmap();

        std::memcpy(mBlobHolder.as<void*>(), minputDataHolder.as<const void*>(), inputData[i]->byteSize());
        inputData[i] = blob;
    }
}

void CommonReferenceTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();

    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (size_t i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto& info = infoIt->second;
        auto blob = inputData[i];

        inferRequest.SetBlob(info->name(), blob);
    }
    inferRequest.Infer();
}

void CommonReferenceTest::Validate() {
    ASSERT_EQ(executableNetwork.GetOutputsInfo().size(), refOutData.size());
    std::vector<InferenceEngine::Blob::Ptr> outputs;
    for (const auto& result : function->get_results()) {
        auto name = ngraph::op::util::create_ie_output_name(result->input_value(0));
        outputs.emplace_back(inferRequest.GetBlob(name));
    }

    ASSERT_EQ(refOutData.size(), outputs.size());
    for (size_t i = 0; i < refOutData.size(); i++) {
        ValidateBlobs(refOutData[i], outputs[i]);
    }
}
void CommonReferenceTest::ValidateBlobs(const InferenceEngine::Blob::Ptr& refBlob, const InferenceEngine::Blob::Ptr& outBlob) {
    ASSERT_TRUE(refBlob != nullptr);
    ASSERT_TRUE(outBlob != nullptr);
    ASSERT_EQ(refBlob->getTensorDesc().getPrecision(), outBlob->getTensorDesc().getPrecision());
    ASSERT_EQ(refBlob->byteSize(), outBlob->byteSize());

    auto mRef = as<InferenceEngine::MemoryBlob>(refBlob);
    IE_ASSERT(mRef);
    const auto refLockMemory = mRef->rmap();
    const auto refBuffer = refLockMemory.as<const std::uint8_t*>();

    auto mOut = as<InferenceEngine::MemoryBlob>(outBlob);
    IE_ASSERT(mOut);
    const auto outLockMemory = mOut->rmap();
    const auto outBuffer = outLockMemory.as<const std::uint8_t*>();

    const auto& precision = refBlob->getTensorDesc().getPrecision();
    switch (precision) {
    case InferenceEngine::Precision::BF16:
        LayerTestsUtils::LayerTestsCommon::Compare<ngraph::bfloat16, ngraph::bfloat16>(
            reinterpret_cast<const ngraph::bfloat16*>(refBuffer), reinterpret_cast<const ngraph::bfloat16*>(outBuffer), refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::FP16:
        LayerTestsUtils::LayerTestsCommon::Compare<ngraph::float16, ngraph::float16>(
            reinterpret_cast<const ngraph::float16*>(refBuffer), reinterpret_cast<const ngraph::float16*>(outBuffer), refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::FP32:
        LayerTestsUtils::LayerTestsCommon::Compare<float, float>(reinterpret_cast<const float*>(refBuffer), reinterpret_cast<const float*>(outBuffer),
                                                                 refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::I8:
        LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(reinterpret_cast<const int8_t*>(refBuffer), reinterpret_cast<const int8_t*>(outBuffer),
                                                                   refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::I16:
        LayerTestsUtils::LayerTestsCommon::Compare<int16_t, int16_t>(reinterpret_cast<const int16_t*>(refBuffer), reinterpret_cast<const int16_t*>(outBuffer),
                                                                     refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::I32:
        LayerTestsUtils::LayerTestsCommon::Compare<int32_t, int32_t>(reinterpret_cast<const int32_t*>(refBuffer), reinterpret_cast<const int32_t*>(outBuffer),
                                                                     refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::I64:
        LayerTestsUtils::LayerTestsCommon::Compare<int64_t, int64_t>(reinterpret_cast<const int64_t*>(refBuffer), reinterpret_cast<const int64_t*>(outBuffer),
                                                                     refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::BOOL:
    case InferenceEngine::Precision::U8:
        LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(reinterpret_cast<const uint8_t*>(refBuffer), reinterpret_cast<const uint8_t*>(outBuffer),
                                                                     refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::U16:
        LayerTestsUtils::LayerTestsCommon::Compare<uint16_t, uint16_t>(reinterpret_cast<const uint16_t*>(refBuffer),
                                                                       reinterpret_cast<const uint16_t*>(outBuffer), refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::U32:
        LayerTestsUtils::LayerTestsCommon::Compare<uint32_t, uint32_t>(reinterpret_cast<const uint32_t*>(refBuffer),
                                                                       reinterpret_cast<const uint32_t*>(outBuffer), refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::U64:
        LayerTestsUtils::LayerTestsCommon::Compare<uint64_t, uint64_t>(reinterpret_cast<const uint64_t*>(refBuffer),
                                                                       reinterpret_cast<const uint64_t*>(outBuffer), refBlob->size(), threshold);
        break;
    case InferenceEngine::Precision::I4:
    case InferenceEngine::Precision::U4:
        LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(reinterpret_cast<const uint8_t*>(refBuffer), reinterpret_cast<const uint8_t*>(outBuffer),
                                                                     refBlob->size() / 2, threshold);
        break;
    case InferenceEngine::Precision::BIN:
        LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(reinterpret_cast<const uint8_t*>(refBuffer), reinterpret_cast<const uint8_t*>(outBuffer),
                                                                     refBlob->size() / 8, threshold);
        break;
    default:
        FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

}  // namespace reference_tests
