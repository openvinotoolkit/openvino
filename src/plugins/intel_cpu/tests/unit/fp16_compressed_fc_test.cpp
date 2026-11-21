// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include <cstddef>
#    include <cstdint>
#    include <memory>
#    include <optional>
#    include <string>
#    include <vector>

#    include "openvino/openvino.hpp"
#    include "openvino/opsets/opset13.hpp"
#    include "openvino/runtime/intel_cpu/properties.hpp"
#    include "utils/precision_support.h"

namespace ov::intel_cpu::test {
namespace {
enum class DecompressionCase {
    Grouped,
    Scalar,
    DifferentGroups,
    RoundedWeights,
    FusedRelu,
    RoundedWeightsKN,
    GroupedWeightsKN,
    Rank3RoundedWeights,
    BatchedRank3RoundedWeights,
    FusedBias
};

void check_fc(ov::element::Type weight_type,
              size_t k,
              bool fractional_zero_point,
              std::optional<bool> compressed,
              bool oversized_zero_point = false,
              int streams = 1,
              float scale = 0.0625F,
              DecompressionCase test_case = DecompressionCase::Grouped) {
    using namespace ov::opset13;
    constexpr size_t n = 17;
    const bool grouped_kn = test_case == DecompressionCase::GroupedWeightsKN;
    const bool kn_layout = test_case == DecompressionCase::RoundedWeightsKN || grouped_kn;
    const bool rounded_weights =
        test_case == DecompressionCase::RoundedWeights || test_case == DecompressionCase::FusedRelu ||
        test_case == DecompressionCase::RoundedWeightsKN || test_case == DecompressionCase::Rank3RoundedWeights ||
        test_case == DecompressionCase::BatchedRank3RoundedWeights || test_case == DecompressionCase::FusedBias;
    const size_t group = rounded_weights ? k : (k == 33 ? 33 : 32);
    const size_t groups = k / group;
    const bool signed_weights = weight_type == ov::element::i4 || weight_type == ov::element::i8;
    std::vector<int32_t> values(n * k);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<int32_t>(i * 7 % 16) - (signed_weights ? 8 : 0);
        if (rounded_weights) {
            values[i] = i % k == 0 ? 3 : (i % k == 1 ? 2 : 0);
            if (kn_layout) {
                values[i] = i / n == 0 ? 3 : (i / n == 1 ? 2 : 0);
            }
        }
    }
    const float zero = rounded_weights ? 0.F
                                       : (signed_weights ? -2.F : 3.F) + (fractional_zero_point ? 0.25F : 0.F) +
                                             (oversized_zero_point ? 32.F : 0.F);
    const ov::Shape weight_shape =
        grouped_kn ? ov::Shape{groups, group, n} : (kn_layout ? ov::Shape{k, n} : ov::Shape{n, groups, group});
    auto weights = Constant::create(weight_type, weight_shape, values);
    auto converted = std::make_shared<Convert>(weights, ov::element::f16);
    const ov::Shape parameter_shape =
        grouped_kn ? ov::Shape{groups, 1, n}
                   : (kn_layout ? ov::Shape{1, n}
                                : (test_case == DecompressionCase::Scalar ? ov::Shape{} : ov::Shape{n, groups, 1}));
    const ov::Shape zero_shape =
        test_case == DecompressionCase::DifferentGroups ? ov::Shape{n, 1, group} : parameter_shape;
    std::vector<float> zero_values(grouped_kn ? groups * n : 1, zero);
    std::vector<float> scale_values(grouped_kn ? groups * n : 1, scale);
    if (grouped_kn) {
        for (size_t i = 0; i < scale_values.size(); ++i) {
            zero_values[i] += static_cast<float>(i % 3);
            scale_values[i] += static_cast<float>(i % 7) * 0.003F;
        }
    }
    auto zero_points = Constant::create(ov::element::f16, zero_shape, zero_values);
    const float rounded_scale = ov::float16(scale);
    auto scales = Constant::create(ov::element::f16, parameter_shape, scale_values);
    auto dequantized = std::make_shared<Multiply>(std::make_shared<Subtract>(converted, zero_points), scales);
    std::shared_ptr<ov::Node> flattened = dequantized;
    if (!kn_layout) {
        flattened =
            std::make_shared<Reshape>(dequantized, Constant::create(ov::element::i64, ov::Shape{2}, {n, k}), false);
    } else if (grouped_kn) {
        flattened =
            std::make_shared<Reshape>(dequantized, Constant::create(ov::element::i64, ov::Shape{2}, {k, n}), false);
    }
    const bool batched_rank3 = test_case == DecompressionCase::BatchedRank3RoundedWeights;
    const bool rank3 = test_case == DecompressionCase::Rank3RoundedWeights || batched_rank3;
    auto input = std::make_shared<Parameter>(ov::element::f16,
                                             batched_rank3 ? ov::PartialShape{-1, 1, static_cast<int64_t>(k)}
                                                           : (rank3 ? ov::PartialShape{1, -1, static_cast<int64_t>(k)}
                                                                    : ov::PartialShape{-1, static_cast<int64_t>(k)}));
    auto matmul = std::make_shared<MatMul>(input, flattened, false, !kn_layout);
    std::shared_ptr<ov::Node> result = matmul;
    if (test_case == DecompressionCase::FusedRelu) {
        result = std::make_shared<Relu>(matmul);
    } else if (test_case == DecompressionCase::FusedBias) {
        result = std::make_shared<Add>(matmul, Constant::create(ov::element::f16, ov::Shape{n}, {0.25F}));
    }
    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       {ov::hint::inference_precision(ov::element::f16),
                                        ov::hint::dynamic_quantization_group_size(32),
                                        ov::inference_num_threads(2),
                                        ov::num_streams(streams)});
    std::vector<ov::InferRequest> requests;
    for (int stream = 0; stream < streams; ++stream) {
        requests.push_back(compiled.create_infer_request());
    }
    for (const size_t m : {1U, 31U, 1U, 2U, 1U}) {
        for (size_t stream = 0; stream < requests.size(); ++stream) {
            ov::Tensor source(ov::element::f16,
                              batched_rank3 ? ov::Shape{m, 1, k} : (rank3 ? ov::Shape{1, m, k} : ov::Shape{m, k}));
            for (size_t i = 0; i < source.get_size(); ++i) {
                source.data<ov::float16>()[i] =
                    ov::float16((static_cast<int>((i * 17 + stream * 11) % 31) - 15) * 0.0625F);
                if (rounded_weights) {
                    source.data<ov::float16>()[i] = ov::float16(i % k == 0 ? 1.F : (i % k == 1 ? -1.F : 0.F));
                    if (test_case == DecompressionCase::FusedRelu && (i / k) % 2) {
                        source.data<ov::float16>()[i] = ov::float16(-static_cast<float>(source.data<ov::float16>()[i]));
                    }
                }
            }
            requests[stream].set_input_tensor(source);
            requests[stream].start_async();
        }
        for (auto& request : requests) {
            request.wait();
            const auto source = request.get_input_tensor();
            const auto output = request.get_output_tensor();
            for (size_t row = 0; row < m; ++row) {
                for (size_t column = 0; column < n; ++column) {
                    float sum = 0;
                    for (size_t channel = 0; channel < k; ++channel) {
                        const size_t index = kn_layout ? channel * n + column : column * k + channel;
                        const size_t parameter_index = grouped_kn ? (channel / group) * n + column : 0;
                        const float reference_zero = ov::float16(zero_values[parameter_index]);
                        const float reference_scale =
                            grouped_kn ? static_cast<float>(ov::float16(scale_values[parameter_index])) : rounded_scale;
                        const ov::float16 difference(static_cast<float>(values[index]) - reference_zero);
                        const ov::float16 value(static_cast<float>(difference) * reference_scale);
                        sum += static_cast<float>(source.data<ov::float16>()[row * k + channel]) *
                               static_cast<float>(value);
                    }
                    const float expected = test_case == DecompressionCase::FusedRelu && sum < 0
                                               ? 0.F
                                               : sum + (test_case == DecompressionCase::FusedBias ? 0.25F : 0.F);
                    ASSERT_EQ(output.data<ov::float16>()[row * n + column], ov::float16(expected));
                }
            }
        }
    }
    size_t fc_count = 0;
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        const auto layer_type = info.find("layerType");
        if (layer_type == info.end() || layer_type->second.as<std::string>() != "FullyConnected") {
            continue;
        }
        ++fc_count;
        if (compressed.has_value()) {
            EXPECT_EQ(node->input_value(1).get_element_type().is_integral_number(), *compressed);
        }
    }
    EXPECT_EQ(fc_count, 1U);
}
}  // namespace

TEST(FP16CompressedFC, DynamicShapesKeepCompressedWeights) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        check_fc(type, 64, false, true);
    }
}

TEST(FP16CompressedFC, UnsupportedDecompressionKeepsMaterializedFallback) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4}) {
        check_fc(type, 33, false, false);
        check_fc(type, 64, true, false);
        check_fc(type, 64, false, false, true);
    }
}

TEST(FP16CompressedFC, MultipleStreamsRetainCompressedExecution) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    check_fc(ov::element::i4, 64, false, true, false, 2);
}

TEST(FP16CompressedFC, DecodePreservesRoundedScaleValues) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4}) {
        check_fc(type, 64, false, true, false, 1, 0.06931F);
    }
}

TEST(FP16CompressedFC, ParameterBroadcastsPreserveInference) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4}) {
        check_fc(type, 64, false, true, false, 1, 0.0625F, DecompressionCase::Scalar);
        // Earlier transformations can normalize these constant broadcasts to a
        // supported compressed representation. Both routes must preserve inference.
        check_fc(type, 2048, false, std::nullopt, false, 1, 0.0625F, DecompressionCase::DifferentGroups);
    }
}

TEST(FP16CompressedFC, PrefillAndDecodePreserveFp16WeightRounding) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        check_fc(type, 64, false, true, false, 1, 1.0009765625F, DecompressionCase::RoundedWeights);
    }
}

TEST(FP16CompressedFC, FusedReluPreservesFp16WeightRounding) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        check_fc(type, 64, false, true, false, 1, 1.0009765625F, DecompressionCase::FusedRelu);
    }
}

TEST(FP16CompressedFC, KNWeightsPreserveFp16Rounding) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        check_fc(type, 64, false, true, false, 1, 1.0009765625F, DecompressionCase::RoundedWeightsKN);
    }
}

TEST(FP16CompressedFC, SharedSquareWeightsPreserveBothLayouts) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    std::vector<int32_t> values(64 * 64, 0);
    values[0] = 3;
    values[64] = 2;
    auto weights = Constant::create(ov::element::u4, ov::Shape{64, 64}, values);
    auto dequantized = std::make_shared<Multiply>(std::make_shared<Convert>(weights, ov::element::f16),
                                                  Constant::create(ov::element::f16, ov::Shape{1, 1}, {1.0009765625F}));
    auto input_node = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{-1, 64});
    auto nk = std::make_shared<MatMul>(input_node, dequantized, false, true);
    auto kn = std::make_shared<MatMul>(input_node, dequantized, false, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{nk, kn}, ov::ParameterVector{input_node});
    ov::Core core;
    const ov::AnyMap config{ov::hint::inference_precision(ov::element::f16),
                            ov::inference_num_threads(2),
                            ov::num_streams(1)};
    auto request = core.compile_model(model, "CPU", config).create_infer_request();
    for (const size_t rows : {1U, 2U, 1U}) {
        ov::Tensor input(ov::element::f16, ov::Shape{rows, 64});
        for (size_t i = 0; i < input.get_size(); ++i) {
            input.data<ov::float16>()[i] = ov::float16(i % 64 == 0 ? 1.F : (i % 64 == 1 ? -1.F : 0.F));
        }
        request.set_input_tensor(input);
        request.infer();
        for (size_t layout = 0; layout < 2; ++layout) {
            const auto output = request.get_output_tensor(layout);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < 64; ++column) {
                    const float expected = layout == 1
                                               ? (column == 0 ? 1.001953125F : 0.F)
                                               : (column == 0 ? 3.00390625F : (column == 1 ? 2.001953125F : 0.F));
                    EXPECT_EQ(output.data<ov::float16>()[row * 64 + column], ov::float16(expected));
                }
            }
        }
    }
}

TEST(FP16CompressedFC, GroupedKNParametersPreserveFp16Rounding) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        check_fc(type, 64, false, true, false, 1, 0.06931F, DecompressionCase::GroupedWeightsKN);
    }
}
TEST(FP16CompressedFC, Rank3PrefillAndDecodePreserveFp16Rounding) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        check_fc(type, 64, false, true, false, 1, 1.0009765625F, DecompressionCase::Rank3RoundedWeights);
    }
}

TEST(FP16CompressedFC, FusedBiasPreservesFp16WeightRounding) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        check_fc(type, 64, false, true, false, 1, 1.0009765625F, DecompressionCase::FusedBias);
    }
}

TEST(FP16CompressedFC, BatchedRank3PrefillAndDecodePreserveFp16Rounding) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        check_fc(type, 64, false, true, false, 1, 1.0009765625F, DecompressionCase::BatchedRank3RoundedWeights);
    }
}
}  // namespace ov::intel_cpu::test
#endif
