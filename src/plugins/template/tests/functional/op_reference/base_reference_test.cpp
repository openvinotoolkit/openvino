// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "base_reference_test.hpp"

#include <gtest/gtest.h>

#include "functional_test_utils/ov_plugin_cache.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace reference_tests {

CommonReferenceTest::CommonReferenceTest() : targetDevice("TEMPLATE") {
    core = test::utils::PluginCache::get().core(targetDevice);
}

void CommonReferenceTest::Exec() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    LoadNetwork();
    FillInputs();
    Infer();
    Validate();
}

void CommonReferenceTest::LoadNetwork() {
    executableNetwork = core->compile_model(function, targetDevice);
}

void CommonReferenceTest::FillInputs() {
    const auto& functionParams = function->get_parameters();
    ASSERT_EQ(functionParams.size(), inputData.size());

    for (size_t i = 0; i < functionParams.size(); i++) {
        const auto& param = functionParams[i];

        ov::Tensor blob;
        if (param->get_partial_shape().is_static()) {
            blob = ov::Tensor(param->get_element_type(), param->get_shape());
        } else {
            blob = ov::Tensor(param->get_element_type(), inputData[i].get_shape());
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
        inferRequest.set_tensor(executableNetwork.input(i), inputData[i]);
    }
    inferRequest.infer();
}

void CommonReferenceTest::Validate() {
    ASSERT_EQ(executableNetwork.outputs().size(), refOutData.size());
    actualOutData.clear();
    for (const auto& output : executableNetwork.outputs()) {
        actualOutData.emplace_back(inferRequest.get_tensor(output));
    }

    ASSERT_EQ(refOutData.size(), actualOutData.size());
    for (size_t i = 0; i < refOutData.size(); i++) {
        ValidateBlobs(refOutData[i], actualOutData[i], i, threshold, abs_threshold, actual_comparision_size);
    }
}

void CommonReferenceTest::ValidateBlobs(const ov::Tensor& refBlob,
                                        const ov::Tensor& outBlob,
                                        const size_t blob_idx,
                                        float threshold,
                                        float abs_threshold,
                                        size_t actual_comparision_size) {
    ASSERT_EQ(refBlob.get_element_type(), outBlob.get_element_type())
        << "Incompatible element type for blob with index " << blob_idx;
    ASSERT_EQ(refBlob.get_byte_size(), outBlob.get_byte_size())
        << "Incorrect byte size for blob with index " << blob_idx;

    if (actual_comparision_size == 0)
        actual_comparision_size = refBlob.get_size();

    const auto& element_type = refBlob.get_element_type();
    switch (element_type) {
    case ov::element::bf16:
        LayerTestsUtils::LayerTestsCommon::Compare<ov::bfloat16, ov::bfloat16>(refBlob.data<const ov::bfloat16>(),
                                                                               outBlob.data<const ov::bfloat16>(),
                                                                               actual_comparision_size,
                                                                               threshold,
                                                                               abs_threshold);
        break;
    case ov::element::f16:
        LayerTestsUtils::LayerTestsCommon::Compare<ov::float16, ov::float16>(refBlob.data<const ov::float16>(),
                                                                             outBlob.data<const ov::float16>(),
                                                                             actual_comparision_size,
                                                                             threshold,
                                                                             abs_threshold);
        break;
    case ov::element::f32:
        LayerTestsUtils::LayerTestsCommon::Compare<float, float>(refBlob.data<const float>(),
                                                                 outBlob.data<const float>(),
                                                                 actual_comparision_size,
                                                                 threshold,
                                                                 abs_threshold);
        break;
    case ov::element::f64:
        LayerTestsUtils::LayerTestsCommon::Compare<double, double>(refBlob.data<const double>(),
                                                                   outBlob.data<const double>(),
                                                                   actual_comparision_size,
                                                                   threshold,
                                                                   abs_threshold);
        break;
    case ov::element::i8:
        LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(refBlob.data<const int8_t>(),
                                                                   outBlob.data<const int8_t>(),
                                                                   actual_comparision_size,
                                                                   threshold,
                                                                   abs_threshold);
        break;
    case ov::element::i16:
        LayerTestsUtils::LayerTestsCommon::Compare<int16_t, int16_t>(refBlob.data<const int16_t>(),
                                                                     outBlob.data<const int16_t>(),
                                                                     actual_comparision_size,
                                                                     threshold,
                                                                     abs_threshold);
        break;
    case ov::element::i32:
        LayerTestsUtils::LayerTestsCommon::Compare<int32_t, int32_t>(refBlob.data<const int32_t>(),
                                                                     outBlob.data<const int32_t>(),
                                                                     actual_comparision_size,
                                                                     threshold,
                                                                     abs_threshold);
        break;
    case ov::element::i64:
        LayerTestsUtils::LayerTestsCommon::Compare<int64_t, int64_t>(refBlob.data<const int64_t>(),
                                                                     outBlob.data<const int64_t>(),
                                                                     actual_comparision_size,
                                                                     threshold,
                                                                     abs_threshold);
        break;
    case ov::element::boolean:
        LayerTestsUtils::LayerTestsCommon::Compare<bool, bool>(refBlob.data<const bool>(),
                                                               outBlob.data<const bool>(),
                                                               actual_comparision_size,
                                                               threshold,
                                                               abs_threshold);
        break;
    case ov::element::u8:
        LayerTestsUtils::LayerTestsCommon::Compare<uint8_t, uint8_t>(refBlob.data<const uint8_t>(),
                                                                     outBlob.data<const uint8_t>(),
                                                                     actual_comparision_size,
                                                                     threshold,
                                                                     abs_threshold);
        break;
    case ov::element::u16:
        LayerTestsUtils::LayerTestsCommon::Compare<uint16_t, uint16_t>(refBlob.data<const uint16_t>(),
                                                                       outBlob.data<const uint16_t>(),
                                                                       actual_comparision_size,
                                                                       threshold,
                                                                       abs_threshold);
        break;
    case ov::element::u32:
        LayerTestsUtils::LayerTestsCommon::Compare<uint32_t, uint32_t>(refBlob.data<const uint32_t>(),
                                                                       outBlob.data<const uint32_t>(),
                                                                       actual_comparision_size,
                                                                       threshold,
                                                                       abs_threshold);
        break;
    case ov::element::u64:
        LayerTestsUtils::LayerTestsCommon::Compare<uint64_t, uint64_t>(refBlob.data<const uint64_t>(),
                                                                       outBlob.data<const uint64_t>(),
                                                                       actual_comparision_size,
                                                                       threshold,
                                                                       abs_threshold);
        break;
    case ov::element::i4:
    case ov::element::u4:
        LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                                   static_cast<const int8_t*>(outBlob.data()),
                                                                   actual_comparision_size / 2,
                                                                   threshold,
                                                                   abs_threshold);
        break;
    case ov::element::u1:
        LayerTestsUtils::LayerTestsCommon::Compare<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                                   static_cast<const int8_t*>(outBlob.data()),
                                                                   actual_comparision_size / 8,
                                                                   threshold,
                                                                   abs_threshold);
        break;
    default:
        FAIL() << "Comparator for " << element_type << " element type isn't supported";
    }
}

}  // namespace reference_tests
