// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/broadcast_u2_zero_point.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"

namespace {

std::shared_ptr<ov::Model> build_dequantize_model(const ov::Shape& weights_shape,
                                                  const ov::Output<ov::Node>& zero_point,
                                                  const ov::element::Type& precision = ov::element::f16) {
    const auto weights = ov::op::v0::Constant::create(ov::element::u2,
                                                      weights_shape,
                                                      std::vector<uint8_t>(ov::shape_size(weights_shape), 2));
    const auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, precision);
    const auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_convert, zero_point);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(subtract)},
                                       ov::ParameterVector{});
}

ov::Output<ov::Node> scalar_zero_point(const float value, const ov::element::Type& type = ov::element::f16) {
    return ov::op::v0::Constant::create(type, ov::Shape{}, {value});
}

std::shared_ptr<ov::Node> zero_point_input(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::Subtract>(op)) {
            return op->get_input_node_shared_ptr(1);
        }
    }
    return nullptr;
}

void run_pass(const std::shared_ptr<ov::Model>& model) {
    ov::npuw::BroadcastU2ZeroPoint pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();
}

void expect_broadcast_to_per_group_u2(const std::shared_ptr<ov::Model>& model) {
    const auto convert = zero_point_input(model);
    ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(convert));
    const auto zero_point = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
    ASSERT_NE(zero_point, nullptr);
    EXPECT_EQ(zero_point->get_element_type(), ov::element::u2);
    EXPECT_EQ(zero_point->get_shape(), (ov::Shape{4, 2, 1}));
    const auto values = zero_point->cast_vector<uint8_t>();
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](uint8_t value) {
        return value == 1;
    }));
    EXPECT_EQ(model->get_results().front()->input_value(0).get_element_type(), ov::element::f16);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_shape(), (ov::Shape{4, 2, 8}));
}

void expect_unchanged(const std::shared_ptr<ov::Model>& model) {
    const auto before = zero_point_input(model);
    run_pass(model);
    EXPECT_EQ(zero_point_input(model), before);
}

}  // namespace

TEST(BroadcastU2ZeroPointPassTest, BroadcastsScalarZeroPointToPerGroupU2) {
    const auto model = build_dequantize_model({4, 2, 8}, scalar_zero_point(1.f));

    run_pass(model);

    expect_broadcast_to_per_group_u2(model);
}

// The graph of ov::decomposition::low_precision_dequantize() for a u8 zero point and f16 scales.
TEST(BroadcastU2ZeroPointPassTest, BroadcastsU8ZeroPointBehindConvert) {
    const auto zero_point =
        std::make_shared<ov::op::v0::Convert>(scalar_zero_point(1.f, ov::element::u8), ov::element::f16);
    const auto model = build_dequantize_model({4, 2, 8}, zero_point);

    run_pass(model);

    expect_broadcast_to_per_group_u2(model);
}

// AsymmZP::CWAI matches only an f32 Constant zero point on the Subtract.
TEST(BroadcastU2ZeroPointPassTest, KeepsF32ZeroPoint) {
    expect_unchanged(build_dequantize_model({4, 2, 8}, scalar_zero_point(1.f, ov::element::f32), ov::element::f32));
}

TEST(BroadcastU2ZeroPointPassTest, KeepsZeroPointThatDoesNotFitInTwoBits) {
    for (const float zero_point : {4.f, -1.f, 0.5f}) {
        expect_unchanged(build_dequantize_model({4, 2, 8}, scalar_zero_point(zero_point)));
    }
}

TEST(BroadcastU2ZeroPointPassTest, KeepsZeroPointOfWeightsWithRankBelowTwo) {
    expect_unchanged(build_dequantize_model({8}, scalar_zero_point(1.f)));
}

TEST(BroadcastU2ZeroPointPassTest, KeepsPerGroupZeroPoint) {
    const auto per_group =
        ov::op::v0::Constant::create(ov::element::u2, ov::Shape{4, 2, 1}, std::vector<uint8_t>(8, 1));
    const auto zero_point = std::make_shared<ov::op::v0::Convert>(per_group, ov::element::f16);

    expect_unchanged(build_dequantize_model({4, 2, 8}, zero_point));
}
