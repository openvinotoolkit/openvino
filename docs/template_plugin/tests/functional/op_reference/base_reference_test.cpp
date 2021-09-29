// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "base_reference_test.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

#include <gtest/gtest.h>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/utils/utils.hpp"

using namespace InferenceEngine;

namespace reference_tests {

CommonReferenceTest::CommonReferenceTest(): targetDevice("TEMPLATE") {
    core = ov::test::PluginCache::get().core(targetDevice);
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
//<<<<<<< HEAD
//    const auto& inputInfo = executableNetwork.GetInputsInfo();
//    const auto& params = function->get_parameters();
//    ASSERT_EQ(params.size(), inputData.size());
//    ASSERT_EQ(inputInfo.size(), inputData.size());
//
//    for (size_t i = 0; i < params.size(); i++) {
//        const auto& param = params[i];
//        const auto infoIt = inputInfo.find(param->get_friendly_name());
//        GTEST_ASSERT_NE(infoIt, inputInfo.cend());
//
//        const auto& info = infoIt->second;
//        InferenceEngine::Blob::Ptr blob;
//        if (!info->getTensorDesc().getDims().empty() && info->getTensorDesc().getDims()[0] != 0) {
//            blob = make_blob_with_precision(info->getTensorDesc());
//        } else {
//            blob = make_plain_blob(info->getTensorDesc().getPrecision(), inputData[i]->getTensorDesc().getDims());
//        }
//        blob->allocate();
//
//        ASSERT_EQ(blob->byteSize(), inputData[i]->byteSize());
//=======
    const auto& functionParams = function->get_parameters();
    ASSERT_EQ(functionParams.size(), inputData.size());

    for (size_t i = 0; i < functionParams.size(); i++) {
        const auto& param = functionParams[i];

        ov::runtime::Tensor blob;
        if (param->get_partial_shape().is_static()) {
            blob = ov::runtime::Tensor(param->get_element_type(), param->get_shape());
        } else {
            blob = ov::runtime::Tensor(param->get_element_type(), inputData[i].get_shape());
        }
        ASSERT_EQ(blob.get_byte_size(), inputData[i].get_byte_size());

        std::memcpy(blob.data(), inputData[i].data(), inputData[i].get_byte_size());
        inputData[i] = blob;
    }
}

void CommonReferenceTest::Infer() {
    inferRequest = executableNetwork.create_infer_request();
    const auto& functionParams = function->get_parameters();

    for (size_t i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        inferRequest.set_tensor(param->get_friendly_name(), inputData[i]);
    }
    inferRequest.infer();
}

void CommonReferenceTest::Validate() {
    ASSERT_EQ(executableNetwork.get_results().size(), refOutData.size());
    std::vector<ov::runtime::Tensor> outputs;
    for (const auto& result : function->get_results()) {
        auto name = ngraph::op::util::create_ie_output_name(result->input_value(0));
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

    const auto& element_type = refBlob.get_element_type();
    switch (element_type) {
    case ov::element::bf16:
        LayerTestsUtils::LayerTestsCommon::Compare<ov::bfloat16, ov::bfloat16>(
            refBlob.data<const ov::bfloat16>(), outBlob.data<const ov::bfloat16>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::f16:
        LayerTestsUtils::LayerTestsCommon::Compare<ov::float16, ov::float16>(
            refBlob.data<const ov::float16>(), outBlob.data<const ov::float16>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::f32:
        LayerTestsUtils::LayerTestsCommon::Compare<float, float>(
            refBlob.data<const float>(), outBlob.data<const float>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::i8:
        LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(
            refBlob.data<const int8_t>(), outBlob.data<const int8_t>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::i16:
        LayerTestsUtils::LayerTestsCommon::Compare<int16_t, int16_t>(
            refBlob.data<const int16_t>(), outBlob.data<const int16_t>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::i32:
        LayerTestsUtils::LayerTestsCommon::Compare<int32_t, int32_t>(
            refBlob.data<const int32_t>(), outBlob.data<const int32_t>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::i64:
        LayerTestsUtils::LayerTestsCommon::Compare<int64_t, int64_t>(
            refBlob.data<const int64_t>(), outBlob.data<const int64_t>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::boolean:
        LayerTestsUtils::LayerTestsCommon::Compare<bool, bool>(
            refBlob.data<const bool>(), outBlob.data<const bool>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::u8:
        LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(
            refBlob.data<const uint8_t>(), outBlob.data<const uint8_t>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::u16:
        LayerTestsUtils::LayerTestsCommon::Compare<uint16_t, uint16_t>(
            refBlob.data<const uint16_t>(), outBlob.data<const uint16_t>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::u32:
        LayerTestsUtils::LayerTestsCommon::Compare<uint32_t, uint32_t>(
            refBlob.data<const uint32_t>(), outBlob.data<const uint32_t>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::u64:
        LayerTestsUtils::LayerTestsCommon::Compare<uint64_t, uint64_t>(
            refBlob.data<const uint64_t>(), outBlob.data<const uint64_t>(),
            refBlob.get_size(), threshold);
        break;
    case ov::element::i4:
    case ov::element::u4:
        LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(
            refBlob.data<const int8_t>(), outBlob.data<const int8_t>(),
            refBlob.get_size() / 2, threshold);
        break;
    case ov::element::u1:
        LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(
            refBlob.data<const int8_t>(), outBlob.data<const int8_t>(),
            refBlob.get_size() / 8, threshold);
        break;
    default:
        FAIL() << "Comparator for " << element_type << " element type isn't supported";
    }
}

}  // namespace reference_tests
