// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "plugin/transformations/dynamic_quantize_fully_connected.hpp"

namespace ov::test::intel_gpu {
namespace {

enum class ZeroPointKind { None, Scalar, Grouped };

struct DynamicQuantizeFCParams {
    size_t input_features;
    size_t output_features;
    size_t weight_groups;
    size_t input_rank;
    uint64_t requested_group_size;
    bool transpose_weights;
    bool transpose_scale;
    bool transpose_zero_point;
    ZeroPointKind zero_point_kind;
    ov::element::Type weight_type;
    bool precompute_reduction;
    bool asymmetric;
    bool use_gs128_for_int8_per_token;
    bool expect_dynamic_quantize;
};

class DynamicQuantizeFullyConnectedTest : public ::testing::TestWithParam<DynamicQuantizeFCParams> {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        ov::Shape input_shape(params.input_rank, 1);
        input_shape.back() = params.input_features;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);

        const ov::Shape weights_shape = params.transpose_weights
                                           ? ov::Shape{params.output_features, params.input_features}
                                           : ov::Shape{params.input_features, params.output_features};
        auto weights = ov::op::v0::Constant::create(params.weight_type, weights_shape, {1});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        const ov::Shape scale_shape = params.transpose_scale
                                          ? ov::Shape{params.output_features, params.weight_groups}
                                          : ov::Shape{params.weight_groups, params.output_features};
        auto scales = ov::op::v0::Constant::create(ov::element::f16, scale_shape, {1});

        std::shared_ptr<ov::intel_gpu::op::FullyConnectedCompressed> fc;
        if (params.zero_point_kind == ZeroPointKind::None) {
            fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                               weights,
                                                                               bias,
                                                                               scales,
                                                                               ov::element::dynamic,
                                                                               params.transpose_weights,
                                                                               params.transpose_scale,
                                                                               params.transpose_scale);
        } else {
            const ov::Shape zero_point_shape =
                params.zero_point_kind == ZeroPointKind::Scalar
                    ? ov::Shape{}
                    : (params.transpose_zero_point ? ov::Shape{params.output_features, params.weight_groups}
                                                   : ov::Shape{params.weight_groups, params.output_features});
            auto zero_points = ov::op::v0::Constant::create(ov::element::u8, zero_point_shape, {0});
            fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input,
                                                                               weights,
                                                                               bias,
                                                                               scales,
                                                                               zero_points,
                                                                               ov::element::dynamic,
                                                                               params.transpose_weights,
                                                                               params.transpose_scale,
                                                                               params.transpose_zero_point);
        }
        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        ov::pass::Manager manager;
        manager.register_pass<ov::intel_gpu::DynamicQuantizeFullyConnected>(params.requested_group_size,
                                                                             params.asymmetric,
                                                                             params.precompute_reduction,
                                                                             params.use_gs128_for_int8_per_token);
        manager.run_passes(model);

        for (const auto& node : model->get_ordered_ops()) {
            if (auto candidate = ov::as_type_ptr<ov::intel_gpu::op::FullyConnectedCompressed>(node)) {
                transformed_fc = std::move(candidate);
                break;
            }
        }
    }

    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::intel_gpu::op::FullyConnectedCompressed> transformed_fc;
};

std::string get_test_case_name(const ::testing::TestParamInfo<DynamicQuantizeFCParams>& info) {
    const auto& p = info.param;
    std::ostringstream name;
    name << "K" << p.input_features << "_N" << p.output_features << "_G" << p.weight_groups << "_GS";
    if (p.requested_group_size == std::numeric_limits<uint64_t>::max()) {
        name << "PerToken";
    } else {
        name << p.requested_group_size;
    }
    name << "_Rank" << p.input_rank << "_TB" << p.transpose_weights << "_TS" << p.transpose_scale << "_TZ"
         << p.transpose_zero_point << "_ZP" << static_cast<int>(p.zero_point_kind) << "_"
         << (p.weight_type == ov::element::i8 ? "i8" : "u8") << "_PR" << p.precompute_reduction << "_Asym"
         << p.asymmetric << "_UseGS128" << p.use_gs128_for_int8_per_token;
    return name.str();
}

TEST_P(DynamicQuantizeFullyConnectedTest, AppliesExpectedQuantizationAndPreservesWeightLayouts) {
    const auto& params = GetParam();
    ASSERT_NE(transformed_fc, nullptr);
    EXPECT_EQ(transformed_fc->get_transpose_b(), params.transpose_weights);
    EXPECT_EQ(transformed_fc->get_transpose_b_scale(), params.transpose_scale);
    if (params.zero_point_kind != ZeroPointKind::None) {
        EXPECT_EQ(transformed_fc->get_transpose_b_zp(), params.transpose_zero_point);
    }

    auto dynamic_quantize =
        ov::as_type_ptr<ov::op::internal::DynamicQuantize>(transformed_fc->input_value(0).get_node_shared_ptr());
    if (!params.expect_dynamic_quantize) {
        EXPECT_EQ(dynamic_quantize, nullptr);
        EXPECT_EQ(transformed_fc->input_value(0).get_element_type(), ov::element::f16);
        return;
    }

    ASSERT_NE(dynamic_quantize, nullptr);
    EXPECT_EQ(dynamic_quantize->get_input_element_type(0), ov::element::f16);

    const uint64_t expected_group_size = params.requested_group_size == std::numeric_limits<uint64_t>::max()
                                             ? (params.use_gs128_for_int8_per_token
                                                    ? 128
                                                    : params.requested_group_size)
                                             : std::min<uint64_t>(params.requested_group_size,
                                                                  params.input_features / params.weight_groups);
    auto expected_group_sizes = std::vector<uint64_t>(params.input_rank, 1);
    expected_group_sizes.back() = expected_group_size;
    EXPECT_EQ(dynamic_quantize->get_group_sizes(), expected_group_sizes);

    const auto expected_quantization_type =
        params.asymmetric && expected_group_size == std::numeric_limits<uint64_t>::max()
            ? ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric
            : ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
    EXPECT_EQ(dynamic_quantize->get_quantization_type(), expected_quantization_type);

    const bool expect_precomputed_reduction = params.precompute_reduction &&
                                              params.zero_point_kind != ZeroPointKind::None &&
                                              expected_group_size != std::numeric_limits<uint64_t>::max();
    EXPECT_EQ(dynamic_quantize->get_attrs().precomputed_reduction, expect_precomputed_reduction);
    if (expect_precomputed_reduction) {
        EXPECT_EQ(transformed_fc->input_value(7).get_node_shared_ptr(), dynamic_quantize);
    } else {
        EXPECT_TRUE(ov::is_type<ov::intel_gpu::op::Placeholder>(transformed_fc->input_value(7).get_node_shared_ptr()));
    }

    EXPECT_EQ(transformed_fc->input_value(0).get_element_type(),
              expected_quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric
                  ? ov::element::u8
                  : ov::element::i8);
}

INSTANTIATE_TEST_SUITE_P(
    DynamicQuantizeScenarios,
    DynamicQuantizeFullyConnectedTest,
    ::testing::Values(
        DynamicQuantizeFCParams{128, 256, 2, 3, 32, true, false, false, ZeroPointKind::Grouped, ov::element::i8, true, false, false, true},
        DynamicQuantizeFCParams{128, 256, 2, 3, 32, true, true, true, ZeroPointKind::Grouped, ov::element::i8, true, false, false, true},
        DynamicQuantizeFCParams{128, 256, 2, 3, 32, false, true, true, ZeroPointKind::Grouped, ov::element::i8, true, false, false, true},
        DynamicQuantizeFCParams{128, 256, 2, 3, 32, false, false, false, ZeroPointKind::Grouped, ov::element::i8, true, false, false, true},
        DynamicQuantizeFCParams{128, 256, 2, 2, 32, true, false, false, ZeroPointKind::None, ov::element::i8, true, false, false, true},
        DynamicQuantizeFCParams{128, 64, 2, 3, std::numeric_limits<uint64_t>::max(), true, false, false, ZeroPointKind::Grouped, ov::element::u8, true, true, false, true},
        DynamicQuantizeFCParams{256, 128, 2, 3, std::numeric_limits<uint64_t>::max(), true, false, false, ZeroPointKind::Grouped, ov::element::i8, true, false, true, true},
        DynamicQuantizeFCParams{128, 64, 2, 3, 16, true, false, false, ZeroPointKind::Grouped, ov::element::i8, true, false, false, false},
        DynamicQuantizeFCParams{120, 64, 2, 3, 32, true, false, false, ZeroPointKind::Grouped, ov::element::i8, true, false, false, false}),
    get_test_case_name);

}  // namespace
}  // namespace ov::test::intel_gpu
