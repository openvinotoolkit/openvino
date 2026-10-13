// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/openvino.hpp"
#include "openvino/reference/avg_pool.hpp"
#include "openvino/reference/fake_quantize.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"

namespace ov::test {
namespace {

using PoolingFqParams = std::tuple<bool, size_t, size_t, bool, ov::element::Type, std::string>;

class AclPoolingFakeQuantizeTest : public testing::TestWithParam<PoolingFqParams> {};

std::shared_ptr<ov::op::v0::Constant> scalar(float value) {
    return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {value});
}

std::shared_ptr<ov::op::v0::FakeQuantize> fake_quantize(const ov::Output<ov::Node>& input,
                                                        float input_low,
                                                        float input_high,
                                                        float output_low,
                                                        float output_high) {
    return std::make_shared<ov::op::v0::FakeQuantize>(input,
                                                      scalar(input_low),
                                                      scalar(input_high),
                                                      scalar(output_low),
                                                      scalar(output_high),
                                                      256);
}

void quantize_reference(const std::vector<float>& input,
                        std::vector<float>& output,
                        const ov::Shape& shape,
                        float input_low,
                        float input_high,
                        float output_low,
                        float output_high) {
    ov::reference::fake_quantize(input.data(),
                                 &input_low,
                                 &input_high,
                                 &output_low,
                                 &output_high,
                                 output.data(),
                                 shape,
                                 {},
                                 {},
                                 {},
                                 {},
                                 256,
                                 ov::op::AutoBroadcastType::NUMPY);
}

TEST_P(AclPoolingFakeQuantizeTest, FoldOutputShiftBeforeCreatingAclQuantizationInfo) {
    const auto& [signed_domain, stride, padding, exclude_pad, precision, layout] = GetParam();
    const ov::Shape input_shape{1, 3, 9, 9};
    const float output_low = signed_domain ? -128.0F : 0.0F;
    const float output_high = signed_domain ? 127.0F : 255.0F;
    // With scale 0.5 the signed input shift is 127.75, while output shift is -128.
    // Reading the former alone truncates it to 127 and cannot recover the signed
    // domain with the legacy >=128 normalization. Folding gives -0.25 instead.
    const float input_low = signed_domain ? -255.5F : 0.5F;
    const float input_high = signed_domain ? 254.5F : 510.5F;
    const ov::Strides strides{stride, stride};
    const ov::Shape pads{padding, padding};

    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto input_fq = fake_quantize(parameter, output_low, output_high, output_low, output_high);
    auto pooling = std::make_shared<ov::op::v1::AvgPool>(input_fq,
                                                         strides,
                                                         pads,
                                                         pads,
                                                         ov::Shape{2, 2},
                                                         exclude_pad,
                                                         ov::op::RoundingType::FLOOR,
                                                         ov::op::PadType::EXPLICIT);
    pooling->set_friendly_name("pooling_shift_regression");
    pooling->get_rt_info()[ov::intel_cpu::InputMemoryFormats::get_type_info_static()] =
        ov::intel_cpu::InputMemoryFormats{"cpu:" + layout};
    pooling->get_rt_info()[ov::intel_cpu::OutputMemoryFormats::get_type_info_static()] =
        ov::intel_cpu::OutputMemoryFormats{"cpu:" + layout};
    auto output_fq = fake_quantize(pooling, input_low, input_high, output_low, output_high);

    // A quantized consumer keeps FQ fused into integer Pooling. Identity weights
    // make every channel observable without cancellation in a channel reduction.
    auto order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(output_fq, order);
    std::vector<int8_t> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
    auto weights = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{3, 3}, identity);
    auto converted_weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    constexpr float weight_scale = 0.125F;
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(converted_weights, scalar(weight_scale));
    auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose, scaled_weights, false, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{parameter});

    ov::Core core;
    auto compiled = core.compile_model(
        model,
        "CPU",
        {ov::hint::inference_precision(precision), ov::num_streams(1), ov::inference_num_threads(1)});
    bool integer_pooling_found = false;
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        if (info.at("layerType").as<std::string>() == "Pooling") {
            integer_pooling_found = true;
            EXPECT_EQ(info.at("runtimePrecision").as<std::string>(), signed_domain ? "i8" : "u8");
            EXPECT_EQ(info.at("outputPrecisions").as<std::string>(), signed_domain ? "i8" : "u8");
            EXPECT_EQ(info.at("primitiveType").as<std::string>().find("acl_"), 0U);
            EXPECT_EQ(info.at("outputLayouts").as<std::string>(), layout == "nchw" ? "abcd" : "acdb");
        }
    }
    ASSERT_TRUE(integer_pooling_found);

    auto request = compiled.create_infer_request();
    auto input_tensor = request.get_input_tensor();
    std::vector<float> input(input_tensor.get_size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = output_low - 32.0F + static_cast<float>((i * 73) % 320) + 0.25F;
    }
    input[0] = 0.0F;
    input[1] = output_low - 32.0F;
    input[2] = output_high + 32.0F;
    std::copy(input.begin(), input.end(), input_tensor.data<float>());
    request.infer();

    std::vector<float> quantized_input(input.size());
    quantize_reference(input, quantized_input, input_shape, output_low, output_high, output_low, output_high);
    const auto pool_shape = pooling->get_output_shape(0);
    std::vector<float> pool_reference(ov::shape_size(pool_shape));
    ov::reference::avg_pool(quantized_input.data(),
                            pool_reference.data(),
                            input_shape,
                            pool_shape,
                            ov::Shape{2, 2},
                            strides,
                            ov::Strides{1, 1},
                            pads,
                            pads,
                            !exclude_pad);
    std::vector<float> output_reference(pool_reference.size());
    quantize_reference(pool_reference, output_reference, pool_shape, input_low, input_high, output_low, output_high);
    const auto actual_tensor = request.get_output_tensor();
    const auto* actual = actual_tensor.data<const float>();
    const auto spatial_size = pool_shape[2] * pool_shape[3];
    ASSERT_EQ(actual_tensor.get_size(), output_reference.size());
    // Integer pooling may round before requantization. Allow one output quantum
    // plus float error; the missing signed shift causes errors near 128 quanta.
    constexpr float tolerance = weight_scale + 1e-5F;
    for (size_t spatial = 0; spatial < spatial_size; ++spatial) {
        for (size_t channel = 0; channel < input_shape[1]; ++channel) {
            const auto expected = output_reference[channel * spatial_size + spatial] * weight_scale;
            ASSERT_NEAR(actual[spatial * input_shape[1] + channel], expected, tolerance)
                << "spatial=" << spatial << ", channel=" << channel;
        }
    }
}

std::vector<PoolingFqParams> pooling_fq_params() {
    std::vector<PoolingFqParams> params;
    for (bool signed_domain : {false, true}) {
        for (size_t stride : {1U, 2U}) {
            for (size_t padding : {0U, 1U}) {
                for (bool exclude_pad : {false, true}) {
                    for (const auto& precision : {ov::element::f16, ov::element::f32}) {
                        for (const std::string layout : {"nchw", "nhwc"}) {
                            // ACL does not support quantized NHWC pooling with
                            // included padding. Keep this regression on its
                            // supported integer executor combinations.
                            if (padding != 0 && !exclude_pad && layout == "nhwc") {
                                continue;
                            }
                            params.emplace_back(signed_domain, stride, padding, exclude_pad, precision, layout);
                        }
                    }
                }
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_ArmPoolingFqShift, AclPoolingFakeQuantizeTest, testing::ValuesIn(pooling_fq_params()));

}  // namespace
}  // namespace ov::test
