// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"

#include <gtest/gtest.h>

#include <numeric>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace {

std::shared_ptr<ov::Model> create_v14_model(const ov::op::RoundingType rounding_type, const bool exclude_pad = true) {
    const auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
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
    const auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto avg_pool_v1 = std::make_shared<v1::AvgPool>(input,
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
    using v0::Concat;
    using v0::Constant;
    using v1::ConvertLike;
    using v1::Pad;
    using v1::Subtract;
    using v3::Broadcast;
    using v3::ShapeOf;

    const auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
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
    const auto avg_pool_v1 = std::make_shared<v1::AvgPool>(pad_node,
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

namespace {
// Kernel 4 with stride 4 over length 7 and end padding 3: CEIL_TORCH yields 2 windows, CEIL adds a third one
// starting in the padding.
std::shared_ptr<ov::Model> create_v14_window_in_padding_model(const ov::PartialShape& shape) {
    const auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
    const auto avg_pool = std::make_shared<ov::op::v14::AvgPool>(input,
                                                                 ov::Strides{4, 4},
                                                                 ov::Shape{0, 0},
                                                                 ov::Shape{3, 3},
                                                                 ov::Shape{4, 4},
                                                                 true,
                                                                 ov::op::RoundingType::CEIL_TORCH);
    return std::make_shared<ov::Model>(avg_pool->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_v1_window_in_padding_model(const ov::PartialShape& shape) {
    const auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
    const auto avg_pool = std::make_shared<v1::AvgPool>(input,
                                                        ov::Strides{4, 4},
                                                        ov::Shape{0, 0},
                                                        ov::Shape{3, 3},
                                                        ov::Shape{4, 4},
                                                        true,
                                                        ov::op::RoundingType::CEIL);
    const auto axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {-2, -1});
    ov::Output<ov::Node> stop;
    if (shape.is_static()) {
        stop = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 2});
    } else {
        const auto shape_of = std::make_shared<v3::ShapeOf>(input, ov::element::i64);
        const auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        const auto lengths = std::make_shared<ov::op::v8::Gather>(shape_of, axes, zero);
        const auto offsets = v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
        const auto shifted = std::make_shared<v1::Add>(lengths, offsets);
        const auto steps = v0::Constant::create(ov::element::i64, ov::Shape{2}, {4, 4});
        stop = std::make_shared<v1::Divide>(shifted, steps);
    }
    const auto start = v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 0});
    const auto step = v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
    const auto slice = std::make_shared<ov::op::v8::Slice>(avg_pool, start, stop, step, axes);
    return std::make_shared<ov::Model>(slice->outputs(), ov::ParameterVector{input});
}
}  // namespace

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_torch_window_in_padding_static) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_window_in_padding_model(ov::PartialShape{1, 3, 7, 7});
    model_ref = create_v1_window_in_padding_model(ov::PartialShape{1, 3, 7, 7});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_torch_window_in_padding_dynamic) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_window_in_padding_model(ov::PartialShape{1, 3, -1, -1});
    model_ref = create_v1_window_in_padding_model(ov::PartialShape{1, 3, -1, -1});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// Length, kernel, stride, pads begin, pads end, exclude pad.
using CeilTorchParams = std::tuple<size_t, size_t, size_t, size_t, size_t, bool>;

class ConvertAvgPool14ToAvgPool1CeilTorch : public testing::TestWithParam<CeilTorchParams> {};

TEST_P(ConvertAvgPool14ToAvgPool1CeilTorch, matches_v14_results) {
    const auto& [length, kernel, stride, pad_begin, pad_end, exclude_pad] = GetParam();
    ov::Tensor data(ov::element::f32, ov::Shape{1, 2, length});
    std::iota(data.data<float>(), data.data<float>() + data.get_size(), 1.0f);
    for (const auto& shape : {ov::PartialShape(data.get_shape()), ov::PartialShape{1, 2, -1}}) {
        const auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        const auto avg_pool = std::make_shared<ov::op::v14::AvgPool>(input,
                                                                     ov::Strides{stride},
                                                                     ov::Shape{pad_begin},
                                                                     ov::Shape{pad_end},
                                                                     ov::Shape{kernel},
                                                                     exclude_pad,
                                                                     ov::op::RoundingType::CEIL_TORCH);
        const auto model = std::make_shared<ov::Model>(avg_pool->outputs(), ov::ParameterVector{input});
        const auto reference = model->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
        manager.run_passes(model);
        for (const auto& node : model->get_ops()) {
            ASSERT_FALSE(ov::is_type<ov::op::v14::AvgPool>(node));
        }
        const auto expected = ov::test::utils::infer_on_template(reference, {data});
        const auto actual = ov::test::utils::infer_on_template(model, {data});
        ASSERT_EQ(actual[0].get_shape(), expected[0].get_shape()) << shape;
        for (size_t i = 0; i < expected[0].get_size(); ++i) {
            ASSERT_NEAR(actual[0].data<float>()[i], expected[0].data<float>()[i], 1e-5f) << shape << " at " << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         ConvertAvgPool14ToAvgPool1CeilTorch,
                         testing::Values(CeilTorchParams{7, 4, 4, 0, 3, true},
                                         CeilTorchParams{5, 2, 2, 1, 1, true},
                                         CeilTorchParams{5, 2, 2, 1, 1, false},
                                         CeilTorchParams{6, 3, 2, 1, 1, true},
                                         CeilTorchParams{10, 8, 2, 0, 1, true},
                                         CeilTorchParams{3, 9, 9, 1, 8, true}));
