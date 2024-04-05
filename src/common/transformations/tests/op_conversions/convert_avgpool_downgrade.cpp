// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset14.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"

namespace {

std::shared_ptr<ov::Model> create_v14_model(const ov::op::RoundingType rounding_type) {
    const auto input = std::make_shared<ov::opset14::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto avg_pool_v14 = std::make_shared<ov::opset14::AvgPool>(input,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     kernel,
                                                                     true,
                                                                     rounding_type,
                                                                     ov::op::PadType::EXPLICIT);

    avg_pool_v14->set_friendly_name("avg_pool_v14");

    return std::make_shared<ov::Model>(avg_pool_v14->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_v1_model(const ov::op::RoundingType rounding_type) {
    const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto avg_pool_v1 = std::make_shared<ov::opset1::AvgPool>(input,
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

std::shared_ptr<ov::Model> create_ceil_torch_workaround_model() {
    const auto input = std::make_shared<ov::opset14::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1};
    ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto padding_begin_node =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{pads_begin.size()}, pads_begin);
    const auto padding_end_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{pads_end.size()}, pads_end);
    const auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto two = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    const auto pads_size = pads_begin.size();
    const auto pads_len = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {pads_size});
    const auto pads_remaining = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 0});

    // gather input spatial dims and prepare for compare as values (in_dim + pad)
    const auto end = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {pads_size + 2});
    const auto dim_idxs = std::make_shared<ov::op::v4::Range>(two, end, one, ov::element::i64);
    const auto shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    const auto gth_in_dims = std::make_shared<ov::op::v8::Gather>(shape, dim_idxs, zero);
    const auto in_left_padded = std::make_shared<ov::op::v1::Add>(gth_in_dims, padding_begin_node);

    // gather output spatial dims and prepare it for compare as values (out_dim - 1) * stride
    const auto ap = std::make_shared<ov::op::v1::AvgPool>(input,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          kernel,
                                                          true,
                                                          ov::op::RoundingType::CEIL);
    const auto shape_of_ap = std::make_shared<ov::op::v3::ShapeOf>(ap, ov::element::i64);
    const auto gth_out_dims = std::make_shared<ov::op::v8::Gather>(shape_of_ap, dim_idxs, zero);
    const auto out_sub_one = std::make_shared<ov::op::v1::Subtract>(gth_out_dims, one);
    const auto stride_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{strides.size()}, strides);
    const auto out_mul_stride = std::make_shared<ov::op::v1::Multiply>(out_sub_one, stride_node);

    // if (in_dim + pad) < ((out_dim - 1) * stride) sliding window in bound use end padding.
    const auto in_gt_out = std::make_shared<ov::op::v1::Greater>(out_mul_stride, in_left_padded);
    const auto selected_pads = std::make_shared<ov::op::v1::Select>(in_gt_out, padding_end_node, zero);

    // apply padding on input clear pads attribute
    const auto pb =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{pads_remaining->output(0), padding_end_node}, 0);
    const auto pe = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{pads_remaining, selected_pads}, 0);
    auto minus_inf =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-std::numeric_limits<float>::infinity()});
    std::shared_ptr<ov::Node> convert_like_node = std::make_shared<ov::op::v1::ConvertLike>(minus_inf, input);
    const auto pad_node =
        std::make_shared<ov::op::v12::Pad>(input, pb, pe, convert_like_node, ov::op::PadMode::CONSTANT);
    std::fill_n(pads_begin.begin(), pads_begin.size(), 0);
    std::fill_n(pads_end.begin(), pads_end.size(), 0);

    const auto avg_pool_v1 = std::make_shared<ov::op::v1::AvgPool>(pad_node,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   kernel,
                                                                   true,
                                                                   ov::op::RoundingType::CEIL,
                                                                   ov::op::PadType::EXPLICIT);

    avg_pool_v1->set_friendly_name("avg_pool_v1_ceil_torch_workaround");

    return std::make_shared<ov::Model>(avg_pool_v1->outputs(), ov::ParameterVector{input});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_torch_to_ceil) {
    model = create_v14_model(ov::op::RoundingType::CEIL_TORCH);
    model_ref = create_ceil_torch_workaround_model();
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    comparator.disable(FunctionsComparator::CmpValues::ACCURACY);
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

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_incorrect_version) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v1_model(ov::op::RoundingType::CEIL);
}
