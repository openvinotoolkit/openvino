// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"

namespace {

std::shared_ptr<ov::Model> create_v14_model(const ov::op::RoundingType rounding_type, const bool exclude_pad = true) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto avg_pool_v14 = std::make_shared<ov::op::v14::AvgPool>(input,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     kernel,
                                                                     exclude_pad,
                                                                     rounding_type,
                                                                     ov::op::PadType::EXPLICIT);

    avg_pool_v14->set_friendly_name("avg_pool_v14");

    return std::make_shared<ov::Model>(avg_pool_v14->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_v1_model(const ov::op::RoundingType rounding_type) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto avg_pool_v1 = std::make_shared<ov::op::v1::AvgPool>(input,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   kernel,
                                                                   true,
                                                                   rounding_type,
                                                                   ov::op::PadType::EXPLICIT);

    avg_pool_v1->set_friendly_name("avg_pool_v1");

    return std::make_shared<ov::Model>(avg_pool_v1->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_exclude_pad_workaround_model() {
    using ov::op::v0::Concat;
    using ov::op::v0::Constant;
    using ov::op::v1::ConvertLike;
    using ov::op::v1::Pad;
    using ov::op::v1::Subtract;
    using ov::op::v3::Broadcast;
    using ov::op::v3::ShapeOf;

    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto zero = Constant::create(ov::element::f32, ov::Shape{}, {0});
    const auto zero_node = std::make_shared<ConvertLike>(zero, input);
    const auto zero_i64 = Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto shape = std::make_shared<ShapeOf>(input, ov::element::i64);
    const auto rank = std::make_shared<ShapeOf>(shape, ov::element::i64);
    const auto pads_begin_node = Constant::create(ov::element::i64, ov::Shape{pads_begin.size()}, pads_begin);
    const auto pads_end_node = Constant::create(ov::element::i64, ov::Shape{pads_end.size()}, pads_end);
    const auto pads_len = Constant::create(ov::element::i64, ov::Shape{}, {pads_begin.size()});
    const auto pads_diff = std::make_shared<Subtract>(rank, pads_len);
    const auto pads_remaining = std::make_shared<Broadcast>(zero_i64, pads_diff);
    const auto pads_begin_v1 =
        std::make_shared<Concat>(ov::OutputVector{std::move(pads_remaining), std::move(pads_begin_node)}, 0);
    const auto pads_end_v1 =
        std::make_shared<Concat>(ov::OutputVector{std::move(pads_remaining), std::move(pads_begin_node)}, 0);
    const auto pad_node =
        std::make_shared<Pad>(input, pads_begin_v1, pads_end_v1, zero_node, ov::op::PadMode::CONSTANT);
    const auto pads_begin_zeros = ov::Shape{0, 0};
    const auto pads_end_zeros = ov::Shape{0, 0};
    const auto avg_pool_v1 = std::make_shared<ov::op::v1::AvgPool>(pad_node,
                                                                   strides,
                                                                   pads_begin_zeros,
                                                                   pads_end_zeros,
                                                                   kernel,
                                                                   false,
                                                                   ov::op::RoundingType::CEIL,
                                                                   ov::op::PadType::EXPLICIT);

    avg_pool_v1->set_friendly_name("avg_pool_v1");

    return std::make_shared<ov::Model>(avg_pool_v1->outputs(), ov::ParameterVector{input});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_torch_to_ceil) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_model(ov::op::RoundingType::CEIL_TORCH);
    model_ref = create_v1_model(ov::op::RoundingType::CEIL);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_to_ceil) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_model(ov::op::RoundingType::CEIL);
    model_ref = create_v1_model(ov::op::RoundingType::CEIL);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_floor_to_floor) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_model(ov::op::RoundingType::FLOOR);
    model_ref = create_v1_model(ov::op::RoundingType::FLOOR);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_torch_to_ceil_no_exclude_pad) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_model(ov::op::RoundingType::CEIL_TORCH, false);
    model_ref = create_exclude_pad_workaround_model();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_torch_to_ceil_exclude_pad) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_model(ov::op::RoundingType::CEIL_TORCH, true);
    model_ref = create_v1_model(ov::op::RoundingType::CEIL);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_incorrect_version) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v1_model(ov::op::RoundingType::CEIL);
}
