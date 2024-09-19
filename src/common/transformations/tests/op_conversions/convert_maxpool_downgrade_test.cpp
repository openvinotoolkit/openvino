// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"

namespace {

std::shared_ptr<ov::Model> create_v14_model(const ov::op::RoundingType rounding_type,
                                            const ov::PartialShape input_shape) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    const ov::Strides strides{2, 2}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto max_pool_v14 = std::make_shared<ov::op::v14::MaxPool>(input,
                                                                     strides,
                                                                     dilations,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     kernel,
                                                                     rounding_type,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::i64,
                                                                     2);

    max_pool_v14->set_friendly_name("max_pool_v14");

    return std::make_shared<ov::Model>(max_pool_v14->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_v8_model(const ov::op::RoundingType rounding_type) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{2, 2}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto max_pool_v8 = std::make_shared<ov::op::v8::MaxPool>(input,
                                                                   strides,
                                                                   dilations,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   kernel,
                                                                   rounding_type,
                                                                   ov::op::PadType::EXPLICIT,
                                                                   ov::element::i64,
                                                                   2);

    max_pool_v8->set_friendly_name("max_pool_v8");

    return std::make_shared<ov::Model>(max_pool_v8->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_ceil_torch_workaround_model(const ov::op::RoundingType rounding_type,
                                                              const bool dynamic_input = false) {
    using ov::op::v0::Concat;
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;
    using ov::op::v1::Add;
    using ov::op::v1::ConvertLike;
    using ov::op::v1::Greater;
    using ov::op::v1::Multiply;
    using ov::op::v1::Select;
    using ov::op::v1::Subtract;
    using ov::op::v12::Pad;
    using ov::op::v3::ShapeOf;
    using ov::op::v4::Range;
    using ov::op::v8::Gather;
    const auto& input_shape =
        dynamic_input
            ? ov::PartialShape{ov::Dimension::dynamic(), 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()}
            : ov::PartialShape{1, 3, 64, 64};
    const auto input = std::make_shared<Parameter>(ov::element::f32, input_shape);
    const ov::Strides strides{2, 2}, dilations{1, 1};
    ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto padding_begin_node = Constant::create(ov::element::i32, ov::Shape{pads_begin.size()}, pads_begin);
    const auto padding_end_node = Constant::create(ov::element::i32, ov::Shape{pads_end.size()}, pads_end);
    const auto zero = Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto one = Constant::create(ov::element::i32, ov::Shape{}, {1});
    const auto two = Constant::create(ov::element::i32, ov::Shape{}, {2});

    const auto pads_size = pads_begin.size();
    const auto pads_len = Constant::create(ov::element::i32, ov::Shape{}, {pads_size});
    const auto pads_remaining = Constant::create(ov::element::i32, ov::Shape{2}, {0, 0});

    // gather input spatial dims and prepare for compare as values (in_dim + pad)
    const auto end = Constant::create(ov::element::i32, ov::Shape{}, {pads_size + 2});
    const auto dim_idxs = std::make_shared<Range>(two, end, one, ov::element::i32);
    const auto shape = std::make_shared<ShapeOf>(input, ov::element::i32);
    const auto gth_in_dims = std::make_shared<Gather>(shape, dim_idxs, zero);
    const auto in_left_padded = std::make_shared<Add>(gth_in_dims, padding_begin_node);

    // gather output spatial dims and prepare it for compare as values (out_dim - 1) * stride
    const auto mp = std::make_shared<ov::op::v8::MaxPool>(input,
                                                          strides,
                                                          dilations,
                                                          pads_begin,
                                                          pads_end,
                                                          kernel,
                                                          ov::op::RoundingType::CEIL);
    const auto shape_of_mp = std::make_shared<ShapeOf>(mp, ov::element::i32);
    const auto gth_out_dims = std::make_shared<Gather>(shape_of_mp, dim_idxs, zero);
    const auto out_sub_one = std::make_shared<Subtract>(gth_out_dims, one);
    const auto stride_node = Constant::create(ov::element::i32, ov::Shape{strides.size()}, strides);
    const auto out_mul_stride = std::make_shared<Multiply>(out_sub_one, stride_node);

    // if (in_dim + pad) > ((out_dim - 1) * stride) sliding window in bound use end padding.
    const auto in_gt_out = std::make_shared<Greater>(in_left_padded, out_mul_stride);
    const auto selected_pads = std::make_shared<Select>(in_gt_out, padding_end_node, zero);

    // apply padding on input clear pads attribute
    const auto pb = std::make_shared<Concat>(ov::OutputVector{pads_remaining, padding_begin_node}, 0);
    const auto pe = std::make_shared<Concat>(ov::OutputVector{pads_remaining, selected_pads}, 0);
    auto minus_inf = Constant::create(ov::element::f32, ov::Shape{}, {-std::numeric_limits<float>::infinity()});
    std::shared_ptr<ov::Node> convert_like_node = std::make_shared<ConvertLike>(minus_inf, input);
    const auto pad_node = std::make_shared<Pad>(input, pb, pe, convert_like_node, ov::op::PadMode::CONSTANT);
    std::fill_n(pads_begin.begin(), pads_begin.size(), 0);
    std::fill_n(pads_end.begin(), pads_end.size(), 0);

    const auto max_pool_v8 = std::make_shared<ov::op::v8::MaxPool>(pad_node,
                                                                   strides,
                                                                   dilations,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   kernel,
                                                                   ov::op::RoundingType::CEIL,
                                                                   ov::op::PadType::EXPLICIT,
                                                                   ov::element::i64,
                                                                   2);
    max_pool_v8->set_friendly_name("max_pool_v8_ceil_torch_workaround");

    return std::make_shared<ov::Model>(max_pool_v8->outputs(), ov::ParameterVector{input});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertMaxPool8ToMaxPool1) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        ov::Strides strides{1}, dilations{1};
        ov::Shape pads_begin{0}, pads_end{0}, kernel{1};
        auto maxpool_8 = std::make_shared<ov::op::v8::MaxPool>(data, strides, dilations, pads_begin, pads_end, kernel);
        auto result = std::make_shared<ov::op::v0::Result>(maxpool_8->output(0));
        model = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::ConvertMaxPool8ToMaxPool1>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        ov::Strides strides{1};
        ov::Shape pads_begin{0}, pads_end{0}, kernel{1};
        auto maxpool_1 = std::make_shared<ov::op::v1::MaxPool>(data, strides, pads_begin, pads_end, kernel);
        auto result = std::make_shared<ov::op::v0::Result>(maxpool_1->output(0));
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_ceil_torch_to_ceil) {
    model = create_v14_model(ov::op::RoundingType::CEIL_TORCH, ov::PartialShape{1, 3, 64, 64});
    model_ref = create_ceil_torch_workaround_model(ov::op::RoundingType::CEIL);
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    comparator.disable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_ceil_torch_to_ceil_dynamic) {
    model = create_v14_model(
        ov::op::RoundingType::CEIL_TORCH,
        ov::PartialShape{ov::Dimension::dynamic(), 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    model_ref = create_ceil_torch_workaround_model(ov::op::RoundingType::CEIL, true);
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    comparator.disable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_ceil_to_ceil) {
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    model = create_v14_model(ov::op::RoundingType::CEIL, ov::PartialShape{1, 3, 64, 64});
    model_ref = create_v8_model(ov::op::RoundingType::CEIL);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_floor_to_floor) {
    model = create_v14_model(ov::op::RoundingType::FLOOR, ov::PartialShape{1, 3, 64, 64});
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    model_ref = create_v8_model(ov::op::RoundingType::FLOOR);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_incorrect_version) {
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    model = create_v8_model(ov::op::RoundingType::CEIL);
}
