// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "plugin/transformations/dynamic_quantize_fully_connected.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

using DynamicQuantize = ov::op::internal::DynamicQuantize;

struct DynamicQuantizeFCParams {
    ov::element::Type weights_type;
    ov::element::Type zp_type;
    bool expect_precomputed_reduction;
};

class DynamicQuantizeFullyConnectedTests : public TransformationTestsF, public WithParamInterface<DynamicQuantizeFCParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<DynamicQuantizeFCParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "weights=" << p.weights_type << "_zp=" << p.zp_type << "_precomputed_reduction=" << p.expect_precomputed_reduction;
        return result.str();
    }

protected:
    // Dynamic quantization group size (128) is larger than the weights group size (64):
    // with precomputed reduction it is adjusted to the weights group size, without it it is kept as is.
    static constexpr size_t K = 1024;
    static constexpr size_t N = 256;
    static constexpr size_t wei_group_size = 64;
    static constexpr uint64_t dyn_quan_group_size = 128;

    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& p = GetParam();
        const ov::Shape decompression_shape{N, K / wei_group_size};
        auto make_weights = [&]() {
            return ov::OutputVector{std::make_shared<ov::op::v0::Constant>(p.weights_type, ov::Shape{N, K}),
                                    std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                    std::make_shared<ov::op::v0::Constant>(ov::element::f16, decompression_shape),
                                    std::make_shared<ov::op::v0::Constant>(p.zp_type, decompression_shape)};
        };
        {
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, K});
            const auto w = make_weights();
            auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, w[0], w[1], w[2], w[3]);
            model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
            manager.register_pass<DynamicQuantizeFullyConnected>(dyn_quan_group_size, false, true);
        }
        {
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, K});
            const auto w = make_weights();
            DynamicQuantize::Attributes attrs;
            attrs.quantization_type = DynamicQuantize::QuantizationType::Symmetric;
            attrs.quantization_dt = ov::element::i8;
            attrs.scale_dt = ov::element::f16;
            attrs.group_sizes = {1, 1, p.expect_precomputed_reduction ? wei_group_size : dyn_quan_group_size};
            if (p.expect_precomputed_reduction) {
                attrs.precomputed_reduction = true;
                attrs.precomputed_reduction_dt = ov::element::i32;
            }
            auto dq = std::make_shared<DynamicQuantize>(input, attrs);
            auto a_zp = std::make_shared<ov::intel_gpu::op::Placeholder>();
            auto precomputed_reduction = p.expect_precomputed_reduction ? dq->output(2) : std::make_shared<ov::intel_gpu::op::Placeholder>()->output(0);
            auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq->output(0),
                                                                                    w[0],
                                                                                    w[1],
                                                                                    w[2],
                                                                                    w[3],
                                                                                    dq->output(1),
                                                                                    a_zp,
                                                                                    precomputed_reduction,
                                                                                    ov::element::f16);
            model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
        }
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

TEST_P(DynamicQuantizeFullyConnectedTests, CompareFunctions) {}

// Precomputed reduction is attached only to 8-bit weights with static zero point
const std::vector<DynamicQuantizeFCParams> dynamic_quantize_fc_params = {
    {ov::element::u8, ov::element::u8, true},
    {ov::element::i8, ov::element::u8, true},
    {ov::element::u4, ov::element::u4, false},
    {ov::element::i4, ov::element::u4, false},
    {ov::element::u3, ov::element::u8, false},
    {ov::element::u2, ov::element::u8, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_TransformationTests_PrecomputedReduction,
                         DynamicQuantizeFullyConnectedTests,
                         ::testing::ValuesIn(dynamic_quantize_fc_params),
                         DynamicQuantizeFullyConnectedTests::get_test_case_name);

constexpr uint64_t per_token_group_size = std::numeric_limits<uint64_t>::max();

struct DynamicQuantizeFCLayoutParams {
    size_t input_features;
    size_t output_features;
    size_t weight_groups;
    size_t input_rank;
    uint64_t requested_group_size;
    bool transpose_weights;
    bool transpose_scale;
    bool transpose_zero_point;
    bool has_zero_point;
    ov::element::Type weights_type;
    bool precompute_reduction;
    bool asymmetric;
    bool use_gs128_for_int8_per_token;
    bool expect_dynamic_quantize;
};

class DynamicQuantizeFullyConnectedLayoutTests : public TransformationTestsF,
                                                 public WithParamInterface<DynamicQuantizeFCLayoutParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<DynamicQuantizeFCLayoutParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "K" << p.input_features << "_N" << p.output_features << "_G" << p.weight_groups << "_GS";
        if (p.requested_group_size == per_token_group_size) {
            result << "PerToken";
        } else {
            result << p.requested_group_size;
        }
        result << "_Rank" << p.input_rank << "_TB" << p.transpose_weights << "_TS" << p.transpose_scale << "_TZ"
               << p.transpose_zero_point << "_ZP" << p.has_zero_point << "_weights=" << p.weights_type << "_PR"
               << p.precompute_reduction << "_Asym" << p.asymmetric << "_UseGS128" << p.use_gs128_for_int8_per_token;
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& p = GetParam();
        auto make_input = [&]() {
            ov::Shape input_shape(p.input_rank, 1);
            input_shape.back() = p.input_features;
            return std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        };
        auto make_decompression_shape = [&](bool transpose) {
            return transpose ? ov::Shape{p.output_features, p.weight_groups} : ov::Shape{p.weight_groups, p.output_features};
        };
        auto make_weights = [&]() {
            const ov::Shape weights_shape = p.transpose_weights ? ov::Shape{p.output_features, p.input_features}
                                                                : ov::Shape{p.input_features, p.output_features};
            ov::OutputVector w{ov::op::v0::Constant::create(p.weights_type, weights_shape, {1}),
                               std::make_shared<ov::intel_gpu::op::Placeholder>(),
                               ov::op::v0::Constant::create(ov::element::f16, make_decompression_shape(p.transpose_scale), {1})};
            if (p.has_zero_point) {
                w.push_back(ov::op::v0::Constant::create(ov::element::u8, make_decompression_shape(p.transpose_zero_point), {0}));
            }
            return w;
        };
        {
            auto input = make_input();
            const auto w = make_weights();
            auto fc = p.has_zero_point
                          ? std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                          w[0],
                                                                                          w[1],
                                                                                          w[2],
                                                                                          w[3],
                                                                                          ov::element::dynamic,
                                                                                          p.transpose_weights,
                                                                                          p.transpose_scale,
                                                                                          p.transpose_zero_point)
                          : std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                                          w[0],
                                                                                          w[1],
                                                                                          w[2],
                                                                                          ov::element::dynamic,
                                                                                          p.transpose_weights,
                                                                                          p.transpose_scale,
                                                                                          p.transpose_zero_point);
            model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
            manager.register_pass<DynamicQuantizeFullyConnected>(p.requested_group_size,
                                                                 p.asymmetric,
                                                                 p.precompute_reduction,
                                                                 p.use_gs128_for_int8_per_token);
        }
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        if (!p.expect_dynamic_quantize) {
            // model_ref defaults to a clone of model, so the transformation must leave the model unchanged
            return;
        }
        {
            auto input = make_input();
            auto w = make_weights();
            if (!p.has_zero_point) {
                w.push_back(std::make_shared<ov::intel_gpu::op::Placeholder>());
            }
            const uint64_t group_size = get_expected_group_size(p);
            const bool is_asymmetric = p.asymmetric && group_size == per_token_group_size;
            const bool expect_precomputed_reduction =
                p.precompute_reduction && p.has_zero_point && group_size != per_token_group_size;

            DynamicQuantize::Attributes attrs;
            attrs.quantization_type =
                is_asymmetric ? DynamicQuantize::QuantizationType::Asymmetric : DynamicQuantize::QuantizationType::Symmetric;
            attrs.quantization_dt = is_asymmetric ? ov::element::u8 : ov::element::i8;
            if (is_asymmetric) {
                attrs.zp_dt = ov::element::u8;
            }
            attrs.scale_dt = ov::element::f16;
            attrs.group_sizes = std::vector<uint64_t>(p.input_rank, 1);
            attrs.group_sizes.back() = group_size;
            if (expect_precomputed_reduction) {
                attrs.precomputed_reduction = true;
                attrs.precomputed_reduction_dt = ov::element::i32;
            }
            auto dq = std::make_shared<DynamicQuantize>(input, attrs);
            auto a_zp = is_asymmetric ? dq->output(2) : std::make_shared<ov::intel_gpu::op::Placeholder>()->output(0);
            auto precomputed_reduction = expect_precomputed_reduction
                                             ? dq->output(is_asymmetric ? 3 : 2)
                                             : std::make_shared<ov::intel_gpu::op::Placeholder>()->output(0);
            auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq->output(0),
                                                                                    w[0],
                                                                                    w[1],
                                                                                    w[2],
                                                                                    w[3],
                                                                                    dq->output(1),
                                                                                    a_zp,
                                                                                    precomputed_reduction,
                                                                                    ov::element::f16,
                                                                                    p.transpose_weights,
                                                                                    p.transpose_scale,
                                                                                    p.transpose_zero_point);
            model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
        }
    }

    static uint64_t get_expected_group_size(const DynamicQuantizeFCLayoutParams& p) {
        if (p.requested_group_size == per_token_group_size) {
            return p.use_gs128_for_int8_per_token ? 128 : per_token_group_size;
        }
        if (p.precompute_reduction && p.has_zero_point) {
            return std::min<uint64_t>(p.requested_group_size, p.input_features / p.weight_groups);
        }
        return p.requested_group_size;
    }
};

TEST_P(DynamicQuantizeFullyConnectedLayoutTests, CompareFunctions) {}

// Weights, scale and zero point transpose flags are independent, so the weights group size
// must be derived from the scale/zero point K axis rather than from the weights layout.
const std::vector<DynamicQuantizeFCLayoutParams> dynamic_quantize_fc_layout_params = {
    // K, N, G, rank, group size, TB, TS, TZ, has ZP, weights, precompute, asym, gs128, expect DQ
    {128, 256, 2, 3, 32, true, false, false, true, ov::element::i8, true, false, false, true},
    {128, 256, 2, 3, 32, true, true, true, true, ov::element::i8, true, false, false, true},
    {128, 256, 2, 3, 32, false, true, true, true, ov::element::i8, true, false, false, true},
    {128, 256, 2, 3, 32, false, false, false, true, ov::element::i8, true, false, false, true},
    {128, 256, 2, 2, 32, true, false, false, false, ov::element::i8, true, false, false, true},
    {128, 64, 2, 3, per_token_group_size, true, false, false, true, ov::element::u8, true, true, false, true},
    {256, 128, 2, 3, per_token_group_size, true, false, false, true, ov::element::i8, true, false, true, true},
    // Unsupported cases: group size below 32 and input features not aligned with group size
    {128, 64, 2, 3, 16, true, false, false, true, ov::element::i8, true, false, false, false},
    {120, 64, 2, 3, 32, true, false, false, true, ov::element::i8, true, false, false, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_TransformationTests_WeightLayouts,
                         DynamicQuantizeFullyConnectedLayoutTests,
                         ::testing::ValuesIn(dynamic_quantize_fc_layout_params),
                         DynamicQuantizeFullyConnectedLayoutTests::get_test_case_name);

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
