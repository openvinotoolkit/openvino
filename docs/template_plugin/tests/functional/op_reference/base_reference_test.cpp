// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "base_reference_test.hpp"

#include <gtest/gtest.h>

#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/utils/utils.hpp"

using namespace InferenceEngine;

namespace reference_tests {

CommonReferenceTest::CommonReferenceTest(): targetDevice("TEMPLATE") {
    // TODO
    // core = PluginCache::get().ie(targetDevice);
}

void CommonReferenceTest::Exec() {
    LoadNetwork();
    FillInputs();
    Infer();
    Validate();
}

void CommonReferenceTest::LoadNetwork() {
    executableNetwork = core->compile_model(function, targetDevice);
}

void CommonReferenceTest::FillInputs() {
    const auto& inputs = function->inputs();
    ASSERT_EQ(inputs.size(), inputData.size());

    for (size_t i = 0; i < inputs.size(); i++) {
        const auto& param = inputs[i];

        ov::runtime::Tensor blob(param->get_element_type(), param->get_shape());
        ASSERT_EQ(blob.get_byte_size(), inputData[i].get_byte_size());

        std::memcpy(blob.data(), inputData[i].data(), inputData[i].get_byte_size());
        inputData[i] = blob;
    }
}

void CommonReferenceTest::Infer() {
    inferRequest = executableNetwork.create_infer_request();
    const auto& execParams = executableNetwork.get_parameters();

    for (size_t i = 0; i < execParams.size(); ++i) {
        const auto& param = execParams[i];
        inferRequest.set_tensor(param->get_friendly_name(), inputData[i]);
    }
    inferRequest.infer();
}

void CommonReferenceTest::Validate() {
    ASSERT_EQ(executableNetwork.get_parameters().size(), refOutData.size());
    std::vector<InferenceEngine::Blob::Ptr> outputs;
    for (const auto& result : function->get_results()) {
        auto name = ov::op::util::create_ie_output_name(result->input_value(0));
        outputs.emplace_back(inferRequest.get_tensor(name));
    }

    ASSERT_EQ(refOutData.size(), outputs.size());
    for (size_t i = 0; i < refOutData.size(); i++) {
        ValidateBlobs(refOutData[i], outputs[i]);
    }
}

void CommonReferenceTest::ValidateBlobs(const ov::runtime::Tensor& refBlob, const ov::runtime::Tensor& outBlob) {
    ASSERT_EQ(refBlob.get_element_type(), outBlob.get_element_type());
    ASSERT_EQ(refBlob.get_byte_size(), outBlob.get_byte_size());

    auto mRef = as<InferenceEngine::MemoryBlob>(refBlob);
    IE_ASSERT(mRef);
    const auto refLockMemory = mRef->rmap();
    const auto refBuffer = refBlob.data();

    auto mOut = as<InferenceEngine::MemoryBlob>(outBlob);
    IE_ASSERT(mOut);
    const auto outLockMemory = mOut->rmap();
    const auto outBuffer = outLockMemory.as<const std::uint8_t*>();

    const auto& precision = refBlob->getTensorDesc().getPrecision();
    switch (precision) {
    case InferenceEngine::Precision::BF16:
        LayerTestsUtils::LayerTestsCommon::Compare<ov::bfloat16, ov::bfloat16>(
            refBlob.data<const ov::bfloat16>(), outBlob.data<const ov::bfloat16>(), refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::FP16:
        LayerTestsUtils::LayerTestsCommon::Compare<ov::float16, ov::float16>(
            refBlob.data<const ov::float16>(), outBlob.data<const ov::float16>(), refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::FP32:
        LayerTestsUtils::LayerTestsCommon::Compare<ov::float16, ov::float16>(
            refBlob.data<const float>(), outBlob.data<const float>(), refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::I8:
        LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(
            refBlob.data<const int8_t>(), outBlob.data<const int8_t>(), refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::I16:
        LayerTestsUtils::LayerTestsCommon::Compare<int16_t, int16_t>(
            refBlob.data<const int16_t>(), outBlob.data<const int16_t>(),
                                                                     refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::I32:
        LayerTestsUtils::LayerTestsCommon::Compare<int32_t, int32_t>(
            refBlob.data<const int32_t>(), outBlob.data<const int32_t>(),
                                                                     refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::I64:
        LayerTestsUtils::LayerTestsCommon::Compare<int64_t, int64_t>(
            refBlob.data<const int64_t>(), outBlob.data<const int64_t>(),
                                                                     refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::BOOL:
    case InferenceEngine::Precision::U8:
        LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(
            refBlob.data<const uint8_t>(), outBlob.data<const uint8_t>(),
                                                                     refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::U16:
        LayerTestsUtils::LayerTestsCommon::Compare<uint16_t, uint16_t>(
            refBlob.data<const uint16_t>(), outBlob.data<const uint16_t>(), refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::U32:
        LayerTestsUtils::LayerTestsCommon::Compare<uint32_t, uint32_t>(
            refBlob.data<const uint32_t>(), outBlob.data<const uint32_t>(), refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::U64:
        LayerTestsUtils::LayerTestsCommon::Compare<uint64_t, uint64_t>(
            refBlob.data<const uint64_t>(), outBlob.data<const uint64_t>(), refBlob.get_size(), threshold);
        break;
    case InferenceEngine::Precision::I4:
    case InferenceEngine::Precision::U4:
        LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(
            refBlob.data<const uint8_t>(), outBlob.data<const uint8_t>(),
                                                                     refBlob.get_size() / 2, threshold);
        break;
    case InferenceEngine::Precision::BIN:
        LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(
            refBlob.data<const uint8_t>(), outBlob.data<const uint8_t>(),
                                                                     refBlob.get_size() / 8, threshold);
        break;
    default:
        FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

}  // namespace reference_tests
