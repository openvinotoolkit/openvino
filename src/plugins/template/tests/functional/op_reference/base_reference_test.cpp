// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "base_reference_test.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/tensor.hpp"
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
        if (param->get_element_type() == ov::element::string) {
            continue;
        }

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
        ValidateBlobs(refOutData[i], actualOutData[i], i, threshold, abs_threshold, legacy_compare);
    }
}

void CommonReferenceTest::ValidateBlobs(const ov::Tensor& refBlob,
                                        const ov::Tensor& outBlob,
                                        const size_t blob_idx,
                                        float threshold,
                                        float abs_threshold,
                                        bool legacy_compare) {
    ASSERT_EQ(refBlob.get_element_type(), outBlob.get_element_type())
        << "Incompatible element type for blob with index " << blob_idx;
    ASSERT_EQ(refBlob.get_byte_size(), outBlob.get_byte_size())
        << "Incorrect byte size for blob with index " << blob_idx;

    // compare() get fundamental element type with element_type_traits firstly and cast data to relative ov type with
    // 'from' types listed below have a fundamental analogue as int8_t, but int8_t is converted only to i8 with from
    std::vector<ov::element::Type> raw_data_comp_only =
        {ov::element::u1, ov::element::u2, ov::element::u3, ov::element::u4, ov::element::u6, ov::element::i4};
    const auto& element_type = refBlob.get_element_type();
    if (!legacy_compare &&
        std::find(raw_data_comp_only.begin(), raw_data_comp_only.end(), element_type) == raw_data_comp_only.end()) {
        switch (element_type) {
        case ov::element::boolean:
        case ov::element::bf16:
        case ov::element::f16:
        case ov::element::f32:
        case ov::element::f64:
        case ov::element::i8:
        case ov::element::i16:
        case ov::element::i32:
        case ov::element::i64:
        case ov::element::u8:
        case ov::element::u16:
        case ov::element::u32:
        case ov::element::u64:
        case ov::element::f8e4m3:
        case ov::element::f8e5m2:
        case ov::element::f8e8m0:
            ov::test::utils::compare(refBlob, outBlob, abs_threshold, threshold);
            break;
        case ov::element::string:
            ov::test::utils::compare_str(refBlob, outBlob);
            break;
        default:
            FAIL() << "Comparator for " << element_type << " element type isn't supported";
        }
        return;
    }

    const auto actual_comparision_size = refBlob.get_size();
    switch (element_type) {
    case ov::element::bf16:
        ov::test::utils::compare_raw_data<ov::bfloat16, ov::bfloat16>(refBlob.data<const ov::bfloat16>(),
                                                                      outBlob.data<const ov::bfloat16>(),
                                                                      actual_comparision_size,
                                                                      threshold,
                                                                      abs_threshold);
        break;
    case ov::element::f16:
        ov::test::utils::compare_raw_data<ov::float16, ov::float16>(refBlob.data<const ov::float16>(),
                                                                    outBlob.data<const ov::float16>(),
                                                                    actual_comparision_size,
                                                                    threshold,
                                                                    abs_threshold);
        break;
    case ov::element::f8e4m3:
        ov::test::utils::compare_raw_data<ov::float8_e4m3, ov::float8_e4m3>(refBlob.data<const ov::float8_e4m3>(),
                                                                            outBlob.data<const ov::float8_e4m3>(),
                                                                            actual_comparision_size,
                                                                            threshold,
                                                                            abs_threshold);
        break;
    case ov::element::f8e5m2:
        ov::test::utils::compare_raw_data<ov::float8_e5m2, ov::float8_e5m2>(refBlob.data<const ov::float8_e5m2>(),
                                                                            outBlob.data<const ov::float8_e5m2>(),
                                                                            actual_comparision_size,
                                                                            threshold,
                                                                            abs_threshold);
        break;
    case ov::element::f8e8m0:
        ov::test::utils::compare_raw_data<ov::float8_e8m0, ov::float8_e8m0>(refBlob.data<const ov::float8_e8m0>(),
                                                                            outBlob.data<const ov::float8_e8m0>(),
                                                                            actual_comparision_size,
                                                                            threshold,
                                                                            abs_threshold);
        break;
    case ov::element::f32:
        ov::test::utils::compare_raw_data<float, float>(refBlob.data<const float>(),
                                                        outBlob.data<const float>(),
                                                        actual_comparision_size,
                                                        threshold,
                                                        abs_threshold);
        break;
    case ov::element::f64:
        ov::test::utils::compare_raw_data<double, double>(refBlob.data<const double>(),
                                                          outBlob.data<const double>(),
                                                          actual_comparision_size,
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::i8:
        ov::test::utils::compare_raw_data<int8_t, int8_t>(refBlob.data<const int8_t>(),
                                                          outBlob.data<const int8_t>(),
                                                          actual_comparision_size,
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::i16:
        ov::test::utils::compare_raw_data<int16_t, int16_t>(refBlob.data<const int16_t>(),
                                                            outBlob.data<const int16_t>(),
                                                            actual_comparision_size,
                                                            threshold,
                                                            abs_threshold);
        break;
    case ov::element::i32:
        ov::test::utils::compare_raw_data<int32_t, int32_t>(refBlob.data<const int32_t>(),
                                                            outBlob.data<const int32_t>(),
                                                            actual_comparision_size,
                                                            threshold,
                                                            abs_threshold);
        break;
    case ov::element::i64:
        ov::test::utils::compare_raw_data<int64_t, int64_t>(refBlob.data<const int64_t>(),
                                                            outBlob.data<const int64_t>(),
                                                            actual_comparision_size,
                                                            threshold,
                                                            abs_threshold);
        break;
    case ov::element::boolean:
        ov::test::utils::compare_raw_data<bool, bool>(refBlob.data<const bool>(),
                                                      outBlob.data<const bool>(),
                                                      actual_comparision_size,
                                                      threshold,
                                                      abs_threshold);
        break;
    case ov::element::u8:
        ov::test::utils::compare_raw_data<uint8_t, uint8_t>(refBlob.data<const uint8_t>(),
                                                            outBlob.data<const uint8_t>(),
                                                            actual_comparision_size,
                                                            threshold,
                                                            abs_threshold);
        break;
    case ov::element::u16:
        ov::test::utils::compare_raw_data<uint16_t, uint16_t>(refBlob.data<const uint16_t>(),
                                                              outBlob.data<const uint16_t>(),
                                                              actual_comparision_size,
                                                              threshold,
                                                              abs_threshold);
        break;
    case ov::element::u32:
        ov::test::utils::compare_raw_data<uint32_t, uint32_t>(refBlob.data<const uint32_t>(),
                                                              outBlob.data<const uint32_t>(),
                                                              actual_comparision_size,
                                                              threshold,
                                                              abs_threshold);
        break;
    case ov::element::u64:
        ov::test::utils::compare_raw_data<uint64_t, uint64_t>(refBlob.data<const uint64_t>(),
                                                              outBlob.data<const uint64_t>(),
                                                              actual_comparision_size,
                                                              threshold,
                                                              abs_threshold);
        break;
    case ov::element::i4:
    case ov::element::u4:
        ov::test::utils::compare_raw_data<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                          static_cast<const int8_t*>(outBlob.data()),
                                                          actual_comparision_size / 2,
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::u1:
        ov::test::utils::compare_raw_data<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                          static_cast<const int8_t*>(outBlob.data()),
                                                          actual_comparision_size / 8,
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::u2:
        ov::test::utils::compare_raw_data<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                          static_cast<const int8_t*>(outBlob.data()),
                                                          actual_comparision_size / 4,
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::u3:
        ov::test::utils::compare_raw_data<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                          static_cast<const int8_t*>(outBlob.data()),
                                                          3 * (actual_comparision_size / 8),
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::u6:
        ov::test::utils::compare_raw_data<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                          static_cast<const int8_t*>(outBlob.data()),
                                                          3 * (actual_comparision_size / 4),
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::nf4:
        ov::test::utils::compare_raw_data<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                          static_cast<const int8_t*>(outBlob.data()),
                                                          actual_comparision_size / 2,
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::f4e2m1:
        ov::test::utils::compare_raw_data<int8_t, int8_t>(static_cast<const int8_t*>(refBlob.data()),
                                                          static_cast<const int8_t*>(outBlob.data()),
                                                          actual_comparision_size / 2,
                                                          threshold,
                                                          abs_threshold);
        break;
    case ov::element::string:
        ov::test::utils::compare_str(refBlob, outBlob);
        break;
    default:
        FAIL() << "Comparator for " << element_type << " element type isn't supported";
    }
}

}  // namespace reference_tests
