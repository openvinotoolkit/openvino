// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pruning.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "mask_attribute.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"
#include "openvino/util/env_util.hpp"
#include "transformations/init_node_info.hpp"

#define VISUALIZE_TESTS_TREE false
#define VISUALIZE_TREE_ROOT  "/tmp/"

using namespace testing;
using namespace ov;
using namespace ov::opset12;

void compare_masks(const Mask& mask, const Mask& ref_mask) {
    ASSERT_EQ(mask.size(), ref_mask.size());
    ASSERT_EQ(mask, ref_mask);
}

void check_mask_is_not_exist(const Mask::Ptr mask) {
    ASSERT_TRUE(!mask);
}

Output<Node> create_constant_with_zeros(const Shape& shape, const Mask& mask) {
    std::vector<double> values(shape_size(shape), 1);
    for (size_t dim = 0; dim < mask.size(); ++dim) {
        for (const auto& dim_value : mask.at(dim)) {
            auto narrow_shape = shape;
            narrow_shape[dim] = 1;
            ov::CoordinateTransformBasic iter(narrow_shape);
            for (auto coord : iter) {
                coord[dim] = dim_value;
                values[coordinate_index(coord, shape)] = 0;
            }
        }
    }
    return std::make_shared<opset10::Constant>(element::f32, shape, values);
}

class DISABLED_TransformationTestsF : public TransformationTestsF {};

class TransformationTestsBoolParamF : public TransformationTestsF, public testing::WithParamInterface<bool> {};

TEST(TransformationTests, InitMasksOI) {
    Shape weights_shape{6, 3, 3, 3};
    auto weights = opset10::Constant::create(element::f32, weights_shape, {0});
    ov::pass::InitConstMask({0, 1}).apply(weights);

    compare_masks(*getMask(weights->output(0)), {{0, 1, 2, 3, 4, 5}, {0, 1, 2}, {}, {}});
}

TEST(TransformationTests, InitMasksOutputChannel) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    std::vector<double> values(shape_size(weights_shape), 1);
    ov::CoordinateTransformBasic iter({6, 1, 3, 3});
    for (auto coord : iter) {
        coord[1] = 1;
        values[coordinate_index(coord, weights_shape)] = 0;
    }

    auto weights = std::make_shared<opset10::Constant>(element::f32, weights_shape, values);
    ov::pass::InitConstMask({1}).apply(weights);

    compare_masks(*getMask(weights->output(0)), {{}, {1}, {}, {}});
}

// TODO: add test init masks with subgraph
TEST(TransformationTests, TestInitMasks) {
    Shape weights_shape{6, 3, 3, 3};
    Shape input_shape{1, 3, 64, 64};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<ov::pass::InitMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), {{1, 2, 3}, {}, {}, {}});
}

TEST(TransformationTests, InitMasksNegative) {
    Shape weights_shape{6, 3, 3, 3};
    auto weights = opset10::Constant::create(element::f32, weights_shape, {0.5});
    ov::pass::InitConstMask({0, 1, 2, 3}).apply(weights);

    compare_masks(*getMask(weights->output(0)), {{}, {}, {}, {}});
}

TEST(TransformationTests, PropagateMasksNegative) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = opset10::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<ov::pass::InitMasks>();
    m.register_pass<ov::pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights->output(0)), {{}, {}, {}, {}});
    compare_masks(*getMask(conv->output(0)), {{}, {}, {}, {}});
}

TEST_F(TransformationTestsF, PropagateMasksBasic) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_shape2{6, 6, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = opset10::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto add_const = create_constant_with_zeros(Shape{1, 6, 1, 1}, {{}, {1, 2, 3, 4, 5}, {}, {}});
    auto add = std::make_shared<opset10::Add>(relu, add_const);

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2}, {}, {}});
    auto sub = std::make_shared<opset10::Subtract>(add, sub_const);

    auto mul_const = create_constant_with_zeros(Shape{1, 6, 1, 1}, {{}, {3}, {}, {}});
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, mul_const);

    auto weights2 = create_constant_with_zeros(weights_shape2, {{1, 2}, {1, 2, 3}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(mul,
                                                        weights2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});

    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);

        auto weights =
            opset10::Constant::create(element::f32,
                                      {weights_shape[0] - 3, weights_shape[1], weights_shape[2], weights_shape[3]},
                                      {0});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
        auto relu = std::make_shared<opset10::Relu>(conv);

        auto add_const = opset10::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1});
        auto add = std::make_shared<opset10::Add>(relu, add_const);

        auto sub_const = opset10::Constant::create(element::f32, Shape{3, 1, 1}, {1});
        auto sub = std::make_shared<opset10::Subtract>(add, sub_const);

        auto mul_const = opset10::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1});
        auto mul = std::make_shared<ov::op::v1::Multiply>(sub, mul_const);

        auto weights2 =
            opset10::Constant::create(element::f32,
                                      {weights_shape2[0], weights_shape2[1] - 3, weights_shape2[2], weights_shape2[3]},
                                      {1});
        auto conv2 = std::make_shared<opset10::Convolution>(mul,
                                                            weights2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksBasic.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add_const), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{1, 2, 3}, {}, {}}));
    compare_masks(*getMask(mul_const), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(weights2.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksDynamicConvolution) {
    PartialShape input_shape{Dimension::dynamic(), 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_shape2{6, 6, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = opset10::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2, 3}, {}, {}});
    auto sub = std::make_shared<opset10::Subtract>(relu, sub_const);

    auto mul_const = create_constant_with_zeros(Shape{6, 1, 1}, {{2}, {}, {}});
    auto mul = std::make_shared<opset10::Subtract>(sub, mul_const);

    auto weights2 = opset10::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(mul,
                                                        weights2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights =
            opset10::Constant::create(element::f32,
                                      {weights_shape[0] - 1, weights_shape[1], weights_shape[2], weights_shape[3]},
                                      {0});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
        auto relu = std::make_shared<opset10::Relu>(conv);

        auto sub_const = create_constant_with_zeros(Shape{5, 1, 1}, {{}, {}, {}});
        auto sub = std::make_shared<opset10::Subtract>(relu, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{5, 1, 1}, {{2}, {}, {}});
        auto mul = std::make_shared<opset10::Subtract>(sub, mul_const);

        auto weights2 =
            opset10::Constant::create(element::f32,
                                      {weights_shape2[0], weights_shape2[1] - 1, weights_shape2[2], weights_shape2[3]},
                                      {0});
        auto conv2 = std::make_shared<opset10::Convolution>(mul,
                                                            weights2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksDynamicConvolution.svg")
            .run_on_model(model);

    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights->output(0)), Mask({{2}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{2}, {}, {}}));
    compare_masks(*getMask(mul_const), Mask({{2}, {}, {}}));
    compare_masks(*getMask(weights2->output(0)), Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, PropagateMasksDynamicReshape) {
    PartialShape input_shape{Dimension::dynamic(), 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_shape2{6, 6, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = opset10::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto reshape =
        std::make_shared<opset10::Reshape>(relu,
                                           opset10::Constant::create(element::i64, Shape{4}, {-1, 6, 64, 64}),
                                           true);

    auto weights2 = opset10::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(reshape,
                                                        weights2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksDynamicReshape.svg").run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::Pruning>();
    m.run_passes(model);

    compare_masks(*getMask(weights->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(reshape), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(weights2->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));
}

TEST(TransformationTests, PropagateMasksDynamicGroupConvolution) {
    PartialShape input_shape{Dimension::dynamic(), 3, 64, 64};
    Shape weights_shape{3, 2, 1, 3, 3};
    Shape weights_shape2{6, 1, 1, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = opset10::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset10::GroupConvolution>(input,
                                                            weights,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2, 3}, {}, {}});
    auto sub = std::make_shared<opset10::Subtract>(relu, sub_const);

    auto mul_const = create_constant_with_zeros(Shape{6, 1, 1}, {{2}, {}, {}});
    auto mul = std::make_shared<opset10::Subtract>(sub, mul_const);

    auto weights2 = opset10::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset10::GroupConvolution>(mul,
                                                             weights2,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));
    auto f = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksDynamicGroupConvolution.svg")
            .run_on_model(f);

    pass::Manager m;
    m.register_pass<ov::pass::InitMasks>();
    m.register_pass<ov::pass::PropagateMasks>();
    m.run_passes(f);
}

TEST(TransformationTests, PropagateMasksEmpty) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_shape2{6, 6, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = opset10::Constant::create(element::f32, weights_shape, {1.});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2, 3}, {}, {}});
    auto sub = std::make_shared<opset10::Subtract>(relu, sub_const);

    auto add_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2}, {}, {}});
    auto add = std::make_shared<opset10::Subtract>(sub, add_const);

    auto weights2 = opset10::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(add,
                                                        weights2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    auto f = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksEmpty.svg").run_on_model(f);

    pass::Manager m;
    m.register_pass<ov::pass::InitMasks>();
    m.register_pass<ov::pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{}, {}, {}}));
    compare_masks(*getMask(add_const), Mask({{}, {}, {}}));
    compare_masks(*getMask(weights2->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));
}

TEST_F(TransformationTestsF, PropagateMaskPassThrough) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_const_1 = create_constant_with_zeros(weights_shape, {{1, 2, 3}, {}, {}, {}});
    weights_const_1.get_node_shared_ptr()->set_friendly_name("weights_1");

    auto conv_1 = std::make_shared<opset10::Convolution>(input,
                                                         weights_const_1,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));
    conv_1->set_friendly_name("conv_1");

    // Adding a couple of PassThrough operations
    auto relu = std::make_shared<opset10::Relu>(conv_1);
    relu->set_friendly_name("relu");

    auto clamp = std::make_shared<opset10::Clamp>(relu, 0, 6);
    clamp->set_friendly_name("clamp");

    auto pads_begin = opset10::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
    auto pads_end = opset10::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
    auto pad = std::make_shared<opset10::Pad>(clamp, pads_begin, pads_end, op::PadMode::CONSTANT);
    auto max_pool =
        std::make_shared<opset10::MaxPool>(pad, Strides{1, 1}, Strides{1, 1}, Shape{0, 0}, Shape{1, 1}, Shape{4, 4});
    max_pool->set_friendly_name("max_pool");

    auto weights2 = opset10::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(max_pool,
                                                        weights2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights_const_1 =
            create_constant_with_zeros({weights_shape[0] - 3, weights_shape[1], weights_shape[2], weights_shape[3]},
                                       {{}, {}, {}, {}});
        weights_const_1.get_node_shared_ptr()->set_friendly_name("weights_1");

        auto conv_1 = std::make_shared<opset10::Convolution>(input,
                                                             weights_const_1,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));
        // Adding a couple of PassThrough operations
        auto relu = std::make_shared<opset10::Relu>(conv_1);

        auto clamp = std::make_shared<opset10::Clamp>(relu, 0, 6);

        auto pads_begin = opset10::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset10::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset10::Pad>(clamp, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto max_pool = std::make_shared<opset10::MaxPool>(pad,
                                                           Strides{1, 1},
                                                           Strides{1, 1},
                                                           Shape{0, 0},
                                                           Shape{1, 1},
                                                           Shape{4, 4});

        auto weights2 =
            opset10::Constant::create(element::f32,
                                      {weight_shape2[0], weight_shape2[1] - 3, weight_shape2[2], weight_shape2[3]},
                                      {0});
        auto conv2 = std::make_shared<opset10::Convolution>(max_pool,
                                                            weights2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMaskPassThrough.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights_const_1.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(clamp->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(max_pool->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, NegativePad12PropagateMaskPassThrough) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_const_1 = create_constant_with_zeros(weights_shape, {{1, 2, 3}, {}, {}, {}});
    weights_const_1.get_node_shared_ptr()->set_friendly_name("weights_1");

    auto conv_1 = std::make_shared<Convolution>(input,
                                                weights_const_1,
                                                Strides(2, 1),
                                                CoordinateDiff(2, 0),
                                                CoordinateDiff(2, 0),
                                                Strides(2, 1));
    conv_1->set_friendly_name("conv_1");

    // Adding a couple of PassThrough operations
    auto relu = std::make_shared<Relu>(conv_1);
    relu->set_friendly_name("relu");

    auto clamp = std::make_shared<Clamp>(relu, 0, 6);
    clamp->set_friendly_name("clamp");

    auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, -1});
    auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, -2});
    auto pad = std::make_shared<ov::op::v12::Pad>(clamp, pads_begin, pads_end, op::PadMode::CONSTANT);
    auto max_pool = std::make_shared<MaxPool>(pad, Strides{1, 1}, Strides{1, 1}, Shape{0, 0}, Shape{1, 1}, Shape{4, 4});
    max_pool->set_friendly_name("max_pool");

    auto weights2 = Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<Convolution>(max_pool,
                                               weights2,
                                               Strides(2, 1),
                                               CoordinateDiff(2, 0),
                                               CoordinateDiff(2, 0),
                                               Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights_const_1 =
            create_constant_with_zeros({weights_shape[0] - 3, weights_shape[1], weights_shape[2], weights_shape[3]},
                                       {{}, {}, {}, {}});
        weights_const_1.get_node_shared_ptr()->set_friendly_name("weights_1");

        auto conv_1 = std::make_shared<Convolution>(input,
                                                    weights_const_1,
                                                    Strides(2, 1),
                                                    CoordinateDiff(2, 0),
                                                    CoordinateDiff(2, 0),
                                                    Strides(2, 1));
        // Adding a couple of PassThrough operations
        auto relu = std::make_shared<Relu>(conv_1);

        auto clamp = std::make_shared<Clamp>(relu, 0, 6);

        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, -2});
        auto pad = std::make_shared<ov::op::v12::Pad>(clamp, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto max_pool =
            std::make_shared<MaxPool>(pad, Strides{1, 1}, Strides{1, 1}, Shape{0, 0}, Shape{1, 1}, Shape{4, 4});

        auto weights2 = Constant::create(element::f32,
                                         {weight_shape2[0], weight_shape2[1] - 3, weight_shape2[2], weight_shape2[3]},
                                         {0});
        auto conv2 = std::make_shared<Convolution>(max_pool,
                                                   weights2,
                                                   Strides(2, 1),
                                                   CoordinateDiff(2, 0),
                                                   CoordinateDiff(2, 0),
                                                   Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMaskPassThrough.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights_const_1.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(clamp->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(max_pool->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksHardDependencies) {
    Shape input_shape{1, 3, 3, 3};

    auto input1 = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    input1->set_friendly_name("input1");

    Shape weights1_shape{6, 3, 3, 3};
    auto weights1 = create_constant_with_zeros(weights1_shape, {{1, 2, 3}, {}, {}, {}});
    weights1.get_node_shared_ptr()->set_friendly_name("weights1");

    auto conv1 = std::make_shared<opset10::Convolution>(input1,
                                                        weights1,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    conv1->set_friendly_name("conv1");

    auto relu = std::make_shared<opset10::Relu>(conv1);
    relu->set_friendly_name("relu");

    auto input2 = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    input2->set_friendly_name("input2");

    Shape weights2_shape{6, 3, 3, 3};
    auto weights2 = create_constant_with_zeros(weights2_shape, {{2, 3}, {}, {}, {}});
    weights2.get_node_shared_ptr()->set_friendly_name("weights2");

    auto conv2 = std::make_shared<opset10::Convolution>(input2,
                                                        weights2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    conv2->set_friendly_name("conv2");

    auto add1 = std::make_shared<opset10::Add>(conv2, conv1);
    add1->set_friendly_name("add1");

    auto reshape =
        std::make_shared<opset10::Reshape>(add1, opset10::Constant::create(element::i64, Shape{2}, {1, 6}), true);
    reshape->set_friendly_name("reshape");

    auto matmul_const = opset10::Constant::create(element::f32, Shape{6, 100}, {1.});
    auto matmul = std::make_shared<opset10::MatMul>(reshape, matmul_const);
    matmul->set_friendly_name("matmul");

    auto add2 = std::make_shared<opset10::Add>(conv2, create_constant_with_zeros({6, 1, 1}, {{2}, {}, {}}));
    add2->set_friendly_name("add2");

    Shape weights_shape3{6, 6, 1, 1};
    auto weights3 = opset10::Constant::create(element::f32, weights_shape3, {0});
    weights3->set_friendly_name("weights3");

    auto conv3 = std::make_shared<opset10::Convolution>(add2,
                                                        weights3,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    conv3->set_friendly_name("conv3");

    model = std::make_shared<Model>(NodeVector{matmul, conv3}, ParameterVector{input1, input2});
    {
        auto input1 = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        input1->set_friendly_name("input1");

        Shape weights1_shape{6, 3, 3, 3};
        auto weights1 =
            create_constant_with_zeros({weights1_shape[0] - 1, weights1_shape[1], weights1_shape[2], weights1_shape[3]},
                                       {{}, {}, {}, {}});
        weights1.get_node_shared_ptr()->set_friendly_name("weights1");

        auto conv1 = std::make_shared<opset10::Convolution>(input1,
                                                            weights1,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        conv1->set_friendly_name("conv1");

        auto relu = std::make_shared<opset10::Relu>(conv1);
        relu->set_friendly_name("relu");

        auto input2 = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        input2->set_friendly_name("input2");

        Shape weights2_shape{6, 3, 3, 3};
        auto weights2 =
            create_constant_with_zeros({weights2_shape[0] - 1, weights2_shape[1], weights2_shape[2], weights2_shape[3]},
                                       {{2, 3}, {}, {}, {}});
        weights2.get_node_shared_ptr()->set_friendly_name("weights2");

        auto conv2 = std::make_shared<opset10::Convolution>(input2,
                                                            weights2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        conv2->set_friendly_name("conv2");

        auto add1 = std::make_shared<opset10::Add>(conv2, conv1);
        add1->set_friendly_name("add1");

        auto reshape =
            std::make_shared<opset10::Reshape>(add1, opset10::Constant::create(element::i64, Shape{2}, {1, 5}), true);
        reshape->set_friendly_name("reshape");

        auto matmul =
            std::make_shared<opset10::MatMul>(reshape, opset10::Constant::create(element::f32, Shape{5, 100}, {1.}));
        matmul->set_friendly_name("matmul");

        auto add2 = std::make_shared<opset10::Add>(conv2, create_constant_with_zeros({5, 1, 1}, {{}, {}, {}}));
        add2->set_friendly_name("add2");

        Shape weights_shape3{6, 6, 1, 1};
        auto weights3 =
            opset10::Constant::create(element::f32,
                                      {weights_shape3[0], weights_shape3[1] - 1, weights_shape3[2], weights_shape3[3]},
                                      {0});
        weights3->set_friendly_name("weights3");

        auto conv3 = std::make_shared<opset10::Convolution>(add2,
                                                            weights3,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        conv3->set_friendly_name("conv3");

        model_ref = std::make_shared<Model>(NodeVector{matmul, conv3}, ParameterVector{input1, input2});
    }

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksHardDependencies.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights1.get_node_shared_ptr()->output(0)), Mask({{2}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)), Mask({{}, {2}, {}, {}}));

    compare_masks(*getMask(weights2.get_node_shared_ptr()->output(0)), Mask({{2}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {2}, {}, {}}));

    compare_masks(*getMask(weights3->output(0)), Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(add1->output(0)), Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(add2->output(0)), Mask({{}, {2}, {}, {}}));

    compare_masks(*getMask(matmul_const->output(0)), Mask({{2}, {}}));
    compare_masks(*getMask(matmul->output(0)), Mask({{}, {}}));

    // TODO: add checks after MatMul/Reshape/Pooling mask propagation is ready
    // compare_masks(*getMask(weights),  Mask({{0, 1, 2, 3, 4, 5}, {}, {}, {}}));
    // compare_masks(*getMask(conv),     Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
    // compare_masks(*getMask(relu),     Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
    // compare_masks(*getMask(weights2), Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
    // compare_masks(*getMask(conv2),    Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksQuantizedGroupConvolution) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weights_group_shape{8, 1, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");

    auto weights1 = create_constant_with_zeros(weights_shape, {{0, 1, 2, 3, 4}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights1,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    auto weights_group = opset10::Constant::create(element::i8, weights_group_shape, {0});
    weights_group->set_friendly_name("weights_group");

    auto convert = std::make_shared<opset10::Convert>(weights_group, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});

    auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{}, {}, {}, {}});
    auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);
    mul->set_friendly_name("mul");

    auto reshape =
        std::make_shared<opset10::Reshape>(mul,
                                           opset10::Constant::create(element::i64, Shape{5}, {8, 1, 1, 3, 3}),
                                           false);

    auto conv_group = std::make_shared<opset10::GroupConvolution>(conv1,
                                                                  reshape,
                                                                  Strides(2, 1),
                                                                  CoordinateDiff(2, 0),
                                                                  CoordinateDiff(2, 0),
                                                                  Strides(2, 1));

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});
    ;
    auto add = std::make_shared<opset10::Add>(conv_group, add_const);
    add->set_friendly_name("add");

    auto weights_2 = opset10::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(add,
                                                        weights_2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);

        auto weights1 =
            create_constant_with_zeros({weights_shape[0] - 5, weights_shape[1], weights_shape[2], weights_shape[3]},
                                       {{}, {}, {}, {}});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        auto weights_group = opset10::Constant::create(
            element::i8,
            {weights_group_shape[0] - 5, weights_group_shape[1], weights_group_shape[2], weights_group_shape[3]},
            {0});

        auto convert = std::make_shared<opset10::Convert>(weights_group, element::f32);

        auto sub_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});

        auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});
        auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);

        auto reshape_const = opset10::Constant::create(element::i64, Shape{5}, {8, 1, 1, 3, 3});

        const auto axis = opset10::Constant::create(ov::element::i8, {}, {0});
        auto dims_to_keep_vec = std::vector<size_t>{2, 3, 4};
        const auto dims_to_keep =
            opset10::Constant::create(reshape_const->get_element_type(), {dims_to_keep_vec.size()}, dims_to_keep_vec);
        const auto reshape_gather = std::make_shared<opset10::Gather>(reshape_const, dims_to_keep, axis);
        const auto reshape_concat = std::make_shared<opset10::Concat>(
            NodeVector{opset10::Constant::create(reshape_const->get_element_type(), {2}, {-1, 1}), reshape_gather},
            0);
        auto reshape = std::make_shared<opset10::Reshape>(mul, reshape_concat, false);

        auto conv_group = std::make_shared<opset10::GroupConvolution>(conv1,
                                                                      reshape,
                                                                      Strides(2, 1),
                                                                      CoordinateDiff(2, 0),
                                                                      CoordinateDiff(2, 0),
                                                                      Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 3, 1, 1}, {{}, {}, {}, {}});
        ;
        auto add = std::make_shared<opset10::Add>(conv_group, add_const);

        auto weights_2 =
            opset10::Constant::create(element::f32,
                                      {weight_shape2[0], weight_shape2[1] - 5, weight_shape2[2], weight_shape2[3]},
                                      {0});
        auto conv2 = std::make_shared<opset10::Convolution>(add,
                                                            weights_2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksQuantizedGroupConvolution.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights1.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_group->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(reshape->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}, {}}));

    compare_masks(*getMask(conv_group->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksQuantizedGroupConvolutionWithShapeOf) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weights_group_shape{8, 1, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");

    auto weights1 = create_constant_with_zeros(weights_shape, {{0, 1, 2, 3}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights1,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    auto weights_group = opset10::Constant::create(element::i8, weights_group_shape, {0});
    weights_group->set_friendly_name("weights_group");

    auto convert = std::make_shared<opset10::Convert>(weights_group, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3}, {}, {}, {}});

    auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});
    auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);
    mul->set_friendly_name("mul");

    auto shape_of = std::make_shared<opset10::ShapeOf>(mul);
    auto axis = opset10::Constant::create(ov::element::i8, {}, {0});
    auto split_lenghts = opset10::Constant::create(ov::element::i8, {2}, {1, -1});
    auto variadic_split = std::make_shared<opset10::VariadicSplit>(shape_of, axis, split_lenghts);
    auto div_const = opset10::Constant::create(ov::element::i64, {1}, {8});
    auto div = std::make_shared<opset10::Divide>(variadic_split->output(0), div_const);
    auto reshape_concat = std::make_shared<opset10::Concat>(
        OutputVector{opset10::Constant::create(shape_of->get_element_type(), {1}, {8})->output(0),
                     div->output(0),
                     variadic_split->output(1)},
        0);

    auto reshape = std::make_shared<opset10::Reshape>(mul, reshape_concat, false);

    auto conv_group = std::make_shared<opset10::GroupConvolution>(conv1,
                                                                  reshape,
                                                                  Strides(2, 1),
                                                                  CoordinateDiff(2, 0),
                                                                  CoordinateDiff(2, 0),
                                                                  Strides(2, 1));

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});
    ;
    auto add = std::make_shared<opset10::Add>(conv_group, add_const);
    add->set_friendly_name("add");

    auto weights_2 = opset10::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(add,
                                                        weights_2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);

        auto weights1 =
            create_constant_with_zeros({weights_shape[0] - 4, weights_shape[1], weights_shape[2], weights_shape[3]},
                                       {{}, {}, {}, {}});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        auto weights_group = opset10::Constant::create(
            element::i8,
            {weights_group_shape[0] - 4, weights_group_shape[1], weights_group_shape[2], weights_group_shape[3]},
            {0});

        auto convert = std::make_shared<opset10::Convert>(weights_group, element::f32);

        auto sub_const = create_constant_with_zeros(Shape{4, 1, 1, 1}, {{}, {}, {}, {}});

        auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{4, 1, 1, 1}, {{}, {}, {}, {}});
        auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);

        auto shape_of = std::make_shared<opset10::ShapeOf>(mul);
        auto axis = opset10::Constant::create(ov::element::i8, {}, {0});
        auto split_lenghts = opset10::Constant::create(ov::element::i8, {2}, {1, -1});
        auto variadic_split = std::make_shared<opset10::VariadicSplit>(shape_of, axis, split_lenghts);
        auto div_const = opset10::Constant::create(ov::element::i64, {1}, {8});
        auto div = std::make_shared<opset10::Divide>(variadic_split->output(0), div_const);

        auto reshape_concat = std::make_shared<opset10::Concat>(
            OutputVector{opset10::Constant::create(shape_of->get_element_type(), {1}, {1})->output(0),
                         div->output(0),
                         variadic_split->output(1)},
            0);

        const auto axis_1 = opset10::Constant::create(ov::element::i8, {}, {0});
        auto dims_to_keep_vec = std::vector<size_t>{2, 3, 4};
        const auto dims_to_keep =
            opset10::Constant::create(reshape_concat->get_element_type(), {dims_to_keep_vec.size()}, dims_to_keep_vec);
        const auto new_reshape_gather = std::make_shared<opset10::Gather>(reshape_concat, dims_to_keep, axis_1);
        const auto new_reshape_concat = std::make_shared<opset10::Concat>(
            NodeVector{opset10::Constant::create(reshape_concat->get_element_type(), {2}, {-1, 1}), new_reshape_gather},
            0);
        auto reshape = std::make_shared<opset10::Reshape>(mul, new_reshape_concat, false);

        auto conv_group = std::make_shared<opset10::GroupConvolution>(conv1,
                                                                      reshape,
                                                                      Strides(2, 1),
                                                                      CoordinateDiff(2, 0),
                                                                      CoordinateDiff(2, 0),
                                                                      Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 4, 1, 1}, {{}, {}, {}, {}});
        ;
        auto add = std::make_shared<opset10::Add>(conv_group, add_const);

        auto weights_2 =
            opset10::Constant::create(element::f32,
                                      {weight_shape2[0], weight_shape2[1] - 4, weight_shape2[2], weight_shape2[3]},
                                      {0});
        auto conv2 = std::make_shared<opset10::Convolution>(add,
                                                            weights_2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksQuantizedGroupConvolutionWithShapeOf.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights1.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)), Mask({{}, {0, 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_group->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));

    compare_masks(*getMask(reshape->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}, {}}));

    compare_masks(*getMask(conv_group->output(0)), Mask({{}, {0, 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)), Mask({{}, {0, 1, 2, 3}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksFakeQuantizePerTensor) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_1 = opset10::Constant::create(element::i8, weights_shape, {0});
    weights_1->set_friendly_name("weights_int8_const");

    auto convert = std::make_shared<opset10::Convert>(weights_1, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});

    auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});
    auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);
    mul->set_friendly_name("mul");

    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        mul,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    conv1->set_friendly_name("conv1");

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});
    ;
    auto add = std::make_shared<opset10::Add>(conv1, add_const);
    add->set_friendly_name("add");

    auto input_low = opset10::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset10::Constant::create(element::f32, Shape{1, 1, 1, 1}, {20});
    auto output_low = opset10::Constant::create(element::f32, Shape{}, {1});
    auto output_high = opset10::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset10::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

    auto weights_2 = opset10::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(fq,
                                                        weights_2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights_1 = opset10::Constant::create(element::i8,
                                                   {
                                                       weights_shape[0] - 5,
                                                       weights_shape[1],
                                                       weights_shape[2],
                                                       weights_shape[3],
                                                   },
                                                   {0});

        auto convert = std::make_shared<opset10::Convert>(weights_1, element::f32);

        auto sub_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});

        auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});
        auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);

        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            mul,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 3, 1, 1}, {{}, {}, {}, {}});
        ;
        auto add = std::make_shared<opset10::Add>(conv1, add_const);

        auto input_low = opset10::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset10::Constant::create(element::f32, Shape{1, 1, 1, 1}, {20});
        auto output_low = opset10::Constant::create(element::f32, Shape{}, {1});
        auto output_high = opset10::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset10::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

        auto weights_2 = opset10::Constant::create(element::f32,
                                                   {
                                                       weight_shape2[0],
                                                       weight_shape2[1] - 5,
                                                       weight_shape2[2],
                                                       weight_shape2[3],
                                                   },
                                                   {0});
        auto conv2 = std::make_shared<opset10::Convolution>(fq,
                                                            weights_2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksFakeQuantizePerTensor.svg")
            .run_on_model(model);

    {
        pass::Manager m;
        // Masks for fq input parammeters didn't saved after
        // ShrinkWeights pass so pruning transformation is splitted
        // on propagation and shrinking passes
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    pass::Manager m;

    compare_masks(*getMask(weights_1->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(conv1->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(fq->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, PropagateMasksFakeQuantizePerTensor1DScale) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_1 = opset10::Constant::create(element::i8, weights_shape, {0});
    weights_1->set_friendly_name("weights_int8_const");

    auto convert = std::make_shared<opset10::Convert>(weights_1, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{1}, {{}});

    auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{1}, {{}});
    auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);
    mul->set_friendly_name("mul");

    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        mul,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    conv1->set_friendly_name("conv1");

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});
    ;
    auto add = std::make_shared<opset10::Add>(conv1, add_const);
    add->set_friendly_name("add");

    auto input_low = opset10::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset10::Constant::create(element::f32, Shape{1, 1, 1, 1}, {20});
    auto output_low = opset10::Constant::create(element::f32, Shape{}, {1});
    auto output_high = opset10::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset10::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

    auto weights_2 = opset10::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(fq,
                                                        weights_2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    auto model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksFakeQuantizePerTensor1DScale.svg")
            .run_on_model(model);

    {
        pass::Manager m;
        m.register_pass<ov::pass::Pruning>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights_1->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv1->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(fq->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));
}

TEST_F(TransformationTestsF, PropagateMasksFakeQuantizePerChannel) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_1 = opset10::Constant::create(element::i8, weights_shape, {0});
    weights_1->set_friendly_name("weights_int8_const");

    auto convert = std::make_shared<opset10::Convert>(weights_1, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});

    auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{}, {}, {}, {}});
    auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);
    mul->set_friendly_name("mul");

    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        mul,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    conv1->set_friendly_name("conv1");

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});
    ;
    auto add = std::make_shared<opset10::Add>(conv1, add_const);
    add->set_friendly_name("add");

    auto input_low = opset10::Constant::create(element::f32, Shape{1, 8, 1, 1}, {0});
    auto input_high = opset10::Constant::create(element::f32, Shape{1, 8, 1, 1}, {20});
    auto output_low = opset10::Constant::create(element::f32, Shape{8, 1, 1}, {1});
    auto output_high = opset10::Constant::create(element::f32, Shape{8, 1, 1}, {10});
    auto fq = std::make_shared<opset10::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

    auto weights_2 = opset10::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset10::Convolution>(fq,
                                                        weights_2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights_1 =
            opset10::Constant::create(element::i8,
                                      {weights_shape[0] - 5, weights_shape[1], weights_shape[2], weights_shape[3]},
                                      {0});

        auto convert = std::make_shared<opset10::Convert>(weights_1, element::f32);

        auto sub_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});

        auto sub = std::make_shared<opset10::Subtract>(convert, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});
        auto mul = std::make_shared<opset10::Multiply>(sub, mul_const);

        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            mul,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 3, 1, 1}, {{}, {}, {}, {}});
        ;
        auto add = std::make_shared<opset10::Add>(conv1, add_const);

        auto input_low = opset10::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0});
        auto input_high = opset10::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20});
        auto output_low = opset10::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1});
        auto output_high = opset10::Constant::create(element::f32, Shape{1, 3, 1, 1}, {10});
        auto fq = std::make_shared<opset10::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

        auto weights_2 =
            opset10::Constant::create(element::f32,
                                      {weight_shape2[0], weight_shape2[1] - 5, weight_shape2[2], weight_shape2[3]},
                                      {0});
        auto conv2 = std::make_shared<opset10::Convolution>(fq,
                                                            weights_2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksFakeQuantizePerChannel.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        // Masks for fq input parammeters didn't saved after
        // ShrinkWeights pass so pruning transformation is splitted
        // on propagation and shrinking passes
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights_1->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(conv1->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(fq->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(fq->input(1).get_source_output()), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(2).get_source_output()), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(3).get_source_output()), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(4).get_source_output()), Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, TestConcatMaskPropagation) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape1{8, 3, 3, 3};
    Shape weights_shape2{16, 3, 3, 3};
    Shape weights_shape3{8, 3, 3, 3};

    Shape weight_shape_out_conv{3, 32, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights_1 = create_constant_with_zeros(weights_shape1, {{0, 1, 2, 3}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights_1,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto weights_2 = create_constant_with_zeros(weights_shape2, {{7, 8, 9, 10}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(input,
                                                        weights_2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto weights_3 = create_constant_with_zeros(weights_shape3, {{4, 5, 6, 7}, {}, {}, {}});
    auto conv3 = std::make_shared<opset10::Convolution>(input,
                                                        weights_3,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto concat =
        std::make_shared<opset10::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

    auto weights_out_conv = create_constant_with_zeros(weight_shape_out_conv, {{}, {}, {}, {}});
    auto conv_out = std::make_shared<opset10::Convolution>(concat,
                                                           weights_out_conv,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv_out}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights_1 =
            create_constant_with_zeros({weights_shape1[0] - 4, weights_shape1[1], weights_shape1[2], weights_shape1[3]},
                                       {{}, {}, {}, {}});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights_1,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        auto weights_2 = create_constant_with_zeros(
            {
                weights_shape2[0] - 4,
                weights_shape2[1],
                weights_shape2[2],
                weights_shape2[3],
            },
            {{}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(input,
                                                            weights_2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        auto weights_3 = create_constant_with_zeros(
            {
                weights_shape3[0] - 4,
                weights_shape3[1],
                weights_shape3[2],
                weights_shape3[3],
            },
            {{}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(input,
                                                            weights_3,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        auto concat =
            std::make_shared<opset10::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

        auto weights_out_conv = create_constant_with_zeros(
            {
                weight_shape_out_conv[0],
                weight_shape_out_conv[1] - 12,
                weight_shape_out_conv[2],
                weight_shape_out_conv[3],
            },
            {{}, {}, {}, {}});
        auto conv_out = std::make_shared<opset10::Convolution>(concat,
                                                               weights_out_conv,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));
        model_ref = std::make_shared<Model>(NodeVector{conv_out}, ParameterVector{input});
    }

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "TestConcatMaskPropagation.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights_1.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)), Mask({{}, {0, 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_2.get_node_shared_ptr()->output(0)), Mask({{7, 8, 9, 10}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {7, 8, 9, 10}, {}, {}}));

    compare_masks(*getMask(weights_3.get_node_shared_ptr()->output(0)), Mask({{4, 5, 6, 7}, {}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)), Mask({{}, {4, 5, 6, 7}, {}, {}}));

    compare_masks(*getMask(concat->output(0)), Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
    compare_masks(*getMask(weights_out_conv.get_node_shared_ptr()->output(0)),
                  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, TestConcatMaskPropagationUp) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape1{8, 3, 3, 3};
    Shape weights_shape2{16, 3, 3, 3};
    Shape weights_shape3{8, 3, 3, 3};

    Shape weight_shape_out_conv{3, 32, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights_1 = create_constant_with_zeros(weights_shape1, {{0, 1, 2, 3, 4, 5}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights_1,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto weights_2 = create_constant_with_zeros(weights_shape2, {{7, 8, 9, 10}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(input,
                                                        weights_2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto weights_3 = create_constant_with_zeros(weights_shape3, {{2, 3, 4, 5, 6, 7}, {}, {}, {}});
    auto conv3 = std::make_shared<opset10::Convolution>(input,
                                                        weights_3,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto concat =
        std::make_shared<opset10::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

    auto add_const =
        create_constant_with_zeros(Shape{1, 32, 1, 1}, {{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}});
    auto add = std::make_shared<opset10::Add>(concat, add_const);

    auto weights_out_conv = create_constant_with_zeros(weight_shape_out_conv, {{}, {}, {}, {}});
    auto conv_out = std::make_shared<opset10::Convolution>(add,
                                                           weights_out_conv,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
    model = std::make_shared<Model>(NodeVector{conv_out}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights_1 = create_constant_with_zeros(
            {
                weights_shape1[0] - 4,
                weights_shape1[1],
                weights_shape1[2],
                weights_shape1[3],
            },
            {{}, {}, {}, {}});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights_1,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        auto weights_2 = create_constant_with_zeros(
            {
                weights_shape2[0] - 4,
                weights_shape2[1],
                weights_shape2[2],
                weights_shape2[3],
            },
            {{}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(input,
                                                            weights_2,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        auto weights_3 = create_constant_with_zeros(
            {
                weights_shape3[0] - 4,
                weights_shape3[1],
                weights_shape3[2],
                weights_shape3[3],
            },
            {{}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(input,
                                                            weights_3,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        auto concat =
            std::make_shared<opset10::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

        auto add_const = create_constant_with_zeros(Shape{1, 20, 1, 1}, {{}, {}, {}, {}});
        auto add = std::make_shared<opset10::Add>(concat, add_const);

        auto weights_out_conv = create_constant_with_zeros(
            {
                weight_shape_out_conv[0],
                weight_shape_out_conv[1] - 12,
                weight_shape_out_conv[2],
                weight_shape_out_conv[3],
            },
            {{}, {}, {}, {}});
        auto conv_out = std::make_shared<opset10::Convolution>(add,
                                                               weights_out_conv,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));

        model_ref = std::make_shared<Model>(NodeVector{conv_out}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "TestConcatMaskPropagationUp.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights_1.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)), Mask({{}, {0, 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_2.get_node_shared_ptr()->output(0)), Mask({{7, 8, 9, 10}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {7, 8, 9, 10}, {}, {}}));

    compare_masks(*getMask(weights_3.get_node_shared_ptr()->output(0)), Mask({{4, 5, 6, 7}, {}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)), Mask({{}, {4, 5, 6, 7}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)),
                  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));

    compare_masks(*getMask(concat->output(0)), Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
    compare_masks(*getMask(weights_out_conv.get_node_shared_ptr()->output(0)),
                  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, TestConcatMaskPropagationUpEmpty) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape1{8, 3, 3, 3};
    Shape weights_shape2{16, 3, 3, 3};
    Shape weights_shape3{8, 3, 3, 3};

    Shape weight_shape_out_conv{3, 32, 3, 3};
    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights_1 = create_constant_with_zeros(weights_shape1, {{0, 1, 2, 3, 4, 5}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights_1,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto weights_2 = create_constant_with_zeros(weights_shape2, {{7, 8, 9, 10}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(input,
                                                        weights_2,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto weights_3 = create_constant_with_zeros(weights_shape3, {{2, 3, 4, 5, 6, 7}, {}, {}, {}});
    auto conv3 = std::make_shared<opset10::Convolution>(input,
                                                        weights_3,
                                                        Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto concat =
        std::make_shared<opset10::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

    auto add_const =
        create_constant_with_zeros(Shape{1, 32, 1, 1}, {{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}});
    auto add = std::make_shared<opset10::Add>(concat, add_const);

    auto f = std::make_shared<Model>(NodeVector{add}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "TestConcatMaskPropagationUpEmpty.svg").run_on_model(f);

    pass::Manager m;
    m.register_pass<ov::pass::InitMasks>();
    m.register_pass<ov::pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights_1.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_2.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_3.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(concat->output(0)), Mask({{}, {}, {}, {}}));
}

TEST_F(TransformationTestsF, PruneConvIsClosingAndInGroup) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto add_const = create_constant_with_zeros(Shape{1, 6, 1, 1}, {{}, {1, 2, 3, 4, 5}, {}, {}});
    auto add = std::make_shared<opset10::Add>(conv, add_const);

    auto conv_1_shape = Shape{weightsShape[0], weightsShape[0], 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(add,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    auto add_1 = std::make_shared<opset10::Add>(conv_1, conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset10::Convolution>(add_1,
                                                           weights_end_conv,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneConvIsClosingAndInGroup.svg").run_on_model(model);
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros(
            {
                weightsShape[0] - 3,
                weightsShape[1],
                weightsShape[2],
                weightsShape[3],
            },
            {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 3, 1, 1}, {{}, {}, {}, {}});
        auto add = std::make_shared<opset10::Add>(conv, add_const);

        auto conv_1_shape = Shape{weightsShape[0] - 3, weightsShape[0] - 3, 1, 1};
        auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset10::Convolution>(add,
                                                             conv_1_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

        auto add_1 = std::make_shared<opset10::Add>(conv_1, conv);

        auto end_conv_shape = Shape{weightsShape[1], weightsShape[0] - 3, 1, 1};
        auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{}, {}, {}, {}});
        auto end_conv = std::make_shared<opset10::Convolution>(add_1,
                                                               weights_end_conv,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));
        model_ref = std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input});
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add_1->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, PruneBranchingStopOp) {
    // Checks case of branching with stop op
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    // Branching stop op
    Shape group_conv_weights_shape{3, 2, 2, 1, 1};
    auto group_conv_weights = opset10::Constant::create(element::f32, group_conv_weights_shape, {0});
    auto group_conv = std::make_shared<opset10::GroupConvolution>(conv,
                                                                  group_conv_weights,
                                                                  Strides(2, 1),
                                                                  CoordinateDiff(2, 0),
                                                                  CoordinateDiff(2, 0),
                                                                  Strides(2, 1));

    auto conv_1_shape = Shape{weightsShape[0], 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(group_conv,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    // Multiply will try to propagate a non zero masks of the conv_1 up
    // and the mask should be invalidated by group conv stop op mask
    auto mul = std::make_shared<opset10::Multiply>(conv_1, conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset10::Convolution>(mul,
                                                           weights_end_conv,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    auto model =
        std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input}, "RestrictedReduceMeanBranching");

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneBranchingStopOp.svg").run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::Pruning>();
    m.run_passes(model);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)), Mask({{}, {}, {}, {}}));
}

TEST(TransformationTests, PruneStopOpUp) {
    // Checks case of branching with stop op
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    // Branching stop op
    Shape group_conv_weights_shape{3, 2, 2, 1, 1};
    auto group_conv_weights = opset10::Constant::create(element::f32, group_conv_weights_shape, {0});
    auto group_conv = std::make_shared<opset10::GroupConvolution>(conv,
                                                                  group_conv_weights,
                                                                  Strides(2, 1),
                                                                  CoordinateDiff(2, 0),
                                                                  CoordinateDiff(2, 0),
                                                                  Strides(2, 1));

    auto conv_1_shape = Shape{weightsShape[0], 6, 1, 1};

    auto mul_const = create_constant_with_zeros(Shape{1, 6, 16, 16}, {{}, {1, 2, 3}, {}, {}});
    auto mul = std::make_shared<opset10::Multiply>(group_conv, mul_const);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset10::Convolution>(mul,
                                                           weights_end_conv,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
    auto model = std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input}, "StopOpUp");

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneStopOpUp.svg").run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::Pruning>();
    m.run_passes(model);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)), Mask({{}, {}, {}, {}}));
}

TEST_F(TransformationTestsF, PruneReducelayerUp) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto reduce_const = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
    auto reduce_mean = std::make_shared<opset10::ReduceMean>(conv, reduce_const, true);

    auto conv_1_shape = Shape{12, 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(reduce_mean,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights =
            create_constant_with_zeros({weightsShape[0] - 3, weightsShape[1], weightsShape[2], weightsShape[3]},
                                       {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

        auto reduce_const = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce_mean = std::make_shared<opset10::ReduceMean>(conv, reduce_const, true);

        auto conv_1_shape = Shape{12, 3, 1, 1};
        auto conv_1_weights =
            create_constant_with_zeros({conv_1_shape[0], conv_1_shape[1], conv_1_shape[2], conv_1_shape[3]},
                                       {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset10::Convolution>(reduce_mean,
                                                             conv_1_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));
        model_ref = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneReducelayerUp.svg").run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::InitMasks>();
    m.register_pass<ov::pass::PropagateMasks>();
    m.run_passes(model);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PruneReduceLayerDown) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto reduce_const = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
    auto reduce_mean = std::make_shared<opset10::ReduceMean>(conv, reduce_const, true);

    auto conv_1_shape = Shape{weightsShape[0], weightsShape[0], 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(reduce_mean,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    auto add_1 = std::make_shared<opset10::Add>(conv_1, conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset10::Convolution>(add_1,
                                                           weights_end_conv,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros(
            {
                weightsShape[0] - 3,
                weightsShape[1],
                weightsShape[2],
                weightsShape[3],
            },
            {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

        auto reduce_const = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce_mean = std::make_shared<opset10::ReduceMean>(conv, reduce_const, true);

        auto conv_1_shape = Shape{weightsShape[0] - 3, weightsShape[0] - 3, 1, 1};
        auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset10::Convolution>(reduce_mean,
                                                             conv_1_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

        auto add_1 = std::make_shared<opset10::Add>(conv_1, conv);

        auto end_conv_shape = Shape{weightsShape[1], weightsShape[0] - 3, 1, 1};
        auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{}, {}, {}, {}});
        auto end_conv = std::make_shared<opset10::Convolution>(add_1,
                                                               weights_end_conv,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));
        model_ref = std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneReduceLayerDown.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(reduce_mean->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add_1->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, PruneStopReducelayerUp) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto reduce_const = opset10::Constant::create(element::i64, Shape{3}, {1, 2, 3});
    auto reduce_mean = std::make_shared<opset10::ReduceMean>(conv, reduce_const, true);

    auto conv_1_shape = Shape{12, 1, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(reduce_mean,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    auto model = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneStopReducelayerUp.svg").run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::Pruning>();
    m.run_passes(model);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));
}

TEST(TransformationTests, PruneStopReduceLayerDown) {
    // Checks case of branching with stop op
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    // Branching stop op
    auto reduce_const = opset10::Constant::create(element::i64, Shape{3}, {1, 2, 3});
    auto reduce_mean = std::make_shared<opset10::ReduceMean>(conv, reduce_const, true);

    auto conv_1_shape = Shape{weightsShape[0], 1, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(reduce_mean,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    // Multiply will try to propagate a non zero masks of the conv_1 up
    // and the mask should be invalidated by reduce_mean stop op mask
    auto mul = std::make_shared<opset10::Multiply>(conv_1, conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset10::Convolution>(mul,
                                                           weights_end_conv,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    auto model =
        std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input}, "RestrictedReduceMeanBranching");

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneStopReduceLayerDown.svg").run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::Pruning>();
    m.run_passes(model);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)), Mask({{}, {}, {}, {}}));
}

TEST_F(TransformationTestsF, MaskPropagationReshapeUp) {
    auto inputShapes = PartialShape{1, 6, 8, 8};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 6, 64, 1});
    auto reshape = std::make_shared<opset10::Reshape>(conv, reshape_const, true);

    auto conv_1_shape = Shape{6, 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(reshape,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros(
            {
                weightsShape[0] - 3,
                weightsShape[1],
                weightsShape[2],
                weightsShape[3],
            },
            {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

        auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 3, 64, 1});
        auto reshape = std::make_shared<opset10::Reshape>(conv, reshape_const, true);

        auto conv_1_shape = Shape{6, 3, 1, 1};
        auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset10::Convolution>(reshape,
                                                             conv_1_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

        model_ref = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeUp.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_P(TransformationTestsBoolParamF, MaskPropagationReshapeUpWithShapeOf) {
    auto inputShapes = PartialShape{1, 6, 8, 8};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    const auto use_shape_of = GetParam();
    Output<Node> reshape_shape_input;
    if (use_shape_of) {
        reshape_shape_input = std::make_shared<opset10::ShapeOf>(conv);
    } else {
        auto shape_of = std::make_shared<opset10::ShapeOf>(conv);
        const auto axis = opset10::Constant::create(ov::element::i8, {}, {0});
        const auto dims_to_keep = opset10::Constant::create(element::i64, {2}, std::vector<int64_t>{2, 3});
        const auto gather = std::make_shared<opset10::Gather>(shape_of, dims_to_keep, axis);
        const auto one_const = opset10::Constant::create(element::i64, {1}, {1});
        const auto minus_one_const = opset10::Constant::create(element::i64, {1}, {-1});
        const auto concat = std::make_shared<opset10::Concat>(NodeVector{one_const, minus_one_const, gather}, 0);
        reshape_shape_input = concat;
    }
    auto reshape = std::make_shared<opset10::Reshape>(conv, reshape_shape_input, true);

    auto conv_1_shape = Shape{6, 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(reshape,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros(
            {
                weightsShape[0] - 3,
                weightsShape[1],
                weightsShape[2],
                weightsShape[3],
            },
            {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

        Output<Node> reshape_shape_input;
        if (use_shape_of) {
            reshape_shape_input = std::make_shared<opset10::ShapeOf>(conv);
        } else {
            auto shape_of = std::make_shared<opset10::ShapeOf>(conv);
            const auto axis = opset10::Constant::create(ov::element::i8, {}, {0});
            const auto dims_to_keep = opset10::Constant::create(element::i64, {2}, std::vector<int64_t>{2, 3});
            const auto gather = std::make_shared<opset10::Gather>(shape_of, dims_to_keep, axis);
            const auto one_const = opset10::Constant::create(element::i64, {1}, {1});
            const auto minus_one_const = opset10::Constant::create(element::i64, {1}, {-1});
            const auto concat = std::make_shared<opset10::Concat>(NodeVector{one_const, minus_one_const, gather}, 0);
            reshape_shape_input = concat;
        }

        auto reshape = std::make_shared<opset10::Reshape>(conv, reshape_shape_input, true);

        auto conv_1_shape = Shape{6, 6, 1, 1};
        auto conv_1_weights = create_constant_with_zeros(
            {
                conv_1_shape[0],
                conv_1_shape[1] - 3,
                conv_1_shape[2],
                conv_1_shape[3],
            },
            {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset10::Convolution>(reshape,
                                                             conv_1_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

        model_ref = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE) {
        const auto postfix = use_shape_of ? "True" : "False";
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeUpWithShapeOf" + postfix + ".svg")
            .run_on_model(model);
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, MaskPropagationReshapeUpShapeSubGraph) {
    auto inputShapes = PartialShape{1, 6, 8, 8};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto shape_of_conv = std::make_shared<opset10::ShapeOf>(conv);
    const auto axis = opset10::Constant::create(ov::element::i8, {}, {0});
    auto dims_to_keep_vec = std::vector<int64_t>{2, 3};
    const auto dims_to_keep = opset10::Constant::create(element::i64, {dims_to_keep_vec.size()}, dims_to_keep_vec);
    const auto gather = std::make_shared<opset10::Gather>(shape_of_conv, dims_to_keep, axis);
    auto dims_to_keep_vec_1 = std::vector<int64_t>{0};
    const auto dims_to_keep_1 =
        opset10::Constant::create(element::i64, {dims_to_keep_vec_1.size()}, dims_to_keep_vec_1);
    const auto gather_1 = std::make_shared<opset10::Gather>(shape_of_conv, dims_to_keep_1, axis);
    const auto concat = std::make_shared<opset10::Concat>(
        NodeVector{gather_1, opset10::Constant::create(element::i64, {1}, {6}), gather},
        0);

    auto reshape = std::make_shared<opset10::Reshape>(conv, concat, true);

    auto conv_1_shape = Shape{6, 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(reshape,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros(
            {
                weightsShape[0] - 3,
                weightsShape[1],
                weightsShape[2],
                weightsShape[3],
            },
            {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

        auto shape_of_conv = std::make_shared<opset10::ShapeOf>(conv);

        const auto axis = opset10::Constant::create(ov::element::i8, {}, {0});
        auto dims_to_keep_vec = std::vector<int64_t>{2, 3};
        const auto dims_to_keep = opset10::Constant::create(element::i64, {dims_to_keep_vec.size()}, dims_to_keep_vec);
        const auto gather = std::make_shared<opset10::Gather>(shape_of_conv, dims_to_keep, axis);
        auto dims_to_keep_vec_1 = std::vector<int64_t>{0};
        const auto dims_to_keep_1 =
            opset10::Constant::create(element::i64, {dims_to_keep_vec_1.size()}, dims_to_keep_vec_1);
        const auto gather_1 = std::make_shared<opset10::Gather>(shape_of_conv, dims_to_keep_1, axis);
        const auto concat = std::make_shared<opset10::Concat>(
            NodeVector{gather_1, opset10::Constant::create(element::i64, {1}, {6}), gather},
            0);

        const auto sub_const = opset10::Constant::create(concat->get_element_type(), {4}, {0, 3, 0, 0});
        const auto sub = std::make_shared<opset10::Subtract>(concat, sub_const);

        auto reshape = std::make_shared<opset10::Reshape>(conv, sub, true);

        auto conv_1_weights = create_constant_with_zeros(
            {
                conv_1_shape[0],
                conv_1_shape[1] - 3,
                conv_1_shape[2],
                conv_1_shape[3],
            },
            {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset10::Convolution>(reshape,
                                                             conv_1_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

        model_ref = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    }

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeUpShapeSubGraph.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, MaskPropagationReshapeExtend) {
    auto inputShapes = Shape{1, 6, 8, 8};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto reshape_const = opset10::Constant::create(element::i64, {3}, {1, -1, 8});
    auto reshape_to = std::make_shared<opset10::Reshape>(conv, reshape_const, true);

    auto mul_const = create_constant_with_zeros({1, 48, 8}, {{}, {5}, {}});
    auto mul = std::make_shared<opset10::Multiply>(reshape_to, mul_const);

    auto shape_of = std::make_shared<opset10::ShapeOf>(conv);
    auto reshape_from = std::make_shared<opset10::Reshape>(mul, shape_of, true);

    auto add_const = create_constant_with_zeros(inputShapes, {{}, {2, 3}, {}, {}});
    auto add = std::make_shared<opset10::Add>(reshape_from, add_const);

    auto conv_1_shape = Shape{6, 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(add,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights =
            create_constant_with_zeros({weightsShape[0] - 2, weightsShape[1], weightsShape[2], weightsShape[3]},
                                       {{1}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

        auto reshape_const = opset10::Constant::create(element::i64, {3}, {1, -1, 8});
        auto reshape_to = std::make_shared<opset10::Reshape>(conv, reshape_const, true);

        auto mul_const = create_constant_with_zeros({1, 32, 8}, {{}, {5}, {}});
        auto mul = std::make_shared<opset10::Multiply>(reshape_to, mul_const);

        auto shape_of = std::make_shared<opset10::ShapeOf>(conv);
        auto reshape_from = std::make_shared<opset10::Reshape>(mul, shape_of, true);

        auto add_const =
            create_constant_with_zeros({inputShapes[0], inputShapes[1] - 2, inputShapes[2], inputShapes[3]},
                                       {{}, {}, {}, {}});
        auto add = std::make_shared<opset10::Add>(reshape_from, add_const);

        auto conv_1_shape = Shape{6, 4, 1, 1};
        auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset10::Convolution>(add,
                                                             conv_1_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

        model_ref = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    }

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeExtend.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {2, 3}, {}, {}}));
    auto reshape_to_mask = Mask(3);
    for (auto i = 16; i < 32; ++i)
        reshape_to_mask.at(1).insert(i);

    compare_masks(*getMask(reshape_to->output(0)), reshape_to_mask);
    compare_masks(*getMask(reshape_from->output(0)), Mask({{}, {2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// Reason: current algo can't process such branching multiply cases
TEST_F(DISABLED_TransformationTestsF, MaskPropagationReshapeDownMul) {
    auto inputShapes = PartialShape{1, 1, 24, 24};
    auto weightsShape = Shape{8, 1, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{}, {}, {}, {}});
    auto first_conv = std::make_shared<opset10::Convolution>(input,
                                                             weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

    auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 8, 576, 1});
    auto reshape = std::make_shared<opset10::Reshape>(first_conv, reshape_const, true);

    auto reshape_conv_weights = create_constant_with_zeros({8, 8, 1, 1}, {{1, 2, 3}, {}, {}, {}});
    auto reshape_conv = std::make_shared<opset10::Convolution>(reshape,
                                                               reshape_conv_weights,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));

    auto reshape_const_1 = opset10::Constant::create(element::i64, Shape{4}, {1, 8, 24, 24});
    auto reshape_1 = std::make_shared<opset10::Reshape>(reshape_conv, reshape_const_1, true);

    auto mul = std::make_shared<opset10::Multiply>(first_conv, reshape_1);

    auto last_conv_weights = create_constant_with_zeros({8, 8, 8, 8}, {{1, 2, 3}, {}, {}, {}});
    auto last_conv = std::make_shared<opset10::Convolution>(mul,
                                                            last_conv_weights,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{last_conv}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros(
            {
                weightsShape[0] - 3,
                weightsShape[1],
                weightsShape[2],
                weightsShape[3],
            },
            {{}, {}, {}, {}});
        auto first_conv = std::make_shared<opset10::Convolution>(input,
                                                                 weights,
                                                                 Strides(2, 1),
                                                                 CoordinateDiff(2, 0),
                                                                 CoordinateDiff(2, 0),
                                                                 Strides(2, 1));

        auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 5, 576, 1});
        auto reshape = std::make_shared<opset10::Reshape>(first_conv, reshape_const, true);

        auto reshape_conv_weights = create_constant_with_zeros({5, 5, 1, 1}, {{}, {}, {}, {}});
        auto reshape_conv = std::make_shared<opset10::Convolution>(reshape,
                                                                   reshape_conv_weights,
                                                                   Strides(2, 1),
                                                                   CoordinateDiff(2, 0),
                                                                   CoordinateDiff(2, 0),
                                                                   Strides(2, 1));

        auto reshape_const_1 = opset10::Constant::create(element::i64, Shape{4}, {1, 5, 24, 24});
        auto reshape_1 = std::make_shared<opset10::Reshape>(reshape_conv, reshape_const_1, true);

        auto mul = std::make_shared<opset10::Multiply>(first_conv, reshape_1);

        auto last_conv_weights = create_constant_with_zeros({8, 5, 8, 8}, {{1, 2, 3}, {}, {}, {}});
        auto last_conv = std::make_shared<opset10::Convolution>(mul,
                                                                last_conv_weights,
                                                                Strides(2, 1),
                                                                CoordinateDiff(2, 0),
                                                                CoordinateDiff(2, 0),
                                                                Strides(2, 1));

        model_ref = std::make_shared<ov::Model>(OutputVector{last_conv}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeDownMul.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(reshape_conv_weights.get_node_shared_ptr()->output(0)),
                  Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(reshape_conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(last_conv_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, MaskPropagationReshapeDownAdd) {
    auto inputShapes = PartialShape{1, 1, 24, 24};
    auto weightsShape = Shape{8, 1, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto first_conv = std::make_shared<opset10::Convolution>(input,
                                                             weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

    auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 8, 576, 1});
    auto reshape = std::make_shared<opset10::Reshape>(first_conv, reshape_const, true);

    auto reshape_conv_weights = create_constant_with_zeros({8, 8, 1, 1}, {{1, 2, 3}, {}, {}, {}});
    auto reshape_conv = std::make_shared<opset10::Convolution>(reshape,
                                                               reshape_conv_weights,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));

    auto reshape_const_1 = opset10::Constant::create(element::i64, Shape{4}, {1, 8, 24, 24});
    auto reshape_1 = std::make_shared<opset10::Reshape>(reshape_conv, reshape_const_1, true);

    auto add = std::make_shared<opset10::Add>(first_conv, reshape_1);

    auto last_conv_weights = create_constant_with_zeros({8, 8, 8, 8}, {{1, 2, 3}, {}, {}, {}});
    auto last_conv = std::make_shared<opset10::Convolution>(add,
                                                            last_conv_weights,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{last_conv}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros(
            {
                weightsShape[0] - 3,
                weightsShape[1],
                weightsShape[2],
                weightsShape[3],
            },
            {{}, {}, {}, {}});
        auto first_conv = std::make_shared<opset10::Convolution>(input,
                                                                 weights,
                                                                 Strides(2, 1),
                                                                 CoordinateDiff(2, 0),
                                                                 CoordinateDiff(2, 0),
                                                                 Strides(2, 1));

        auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 5, 576, 1});
        auto reshape = std::make_shared<opset10::Reshape>(first_conv, reshape_const, true);

        auto reshape_conv_weights = create_constant_with_zeros({5, 5, 1, 1}, {{}, {}, {}, {}});
        auto reshape_conv = std::make_shared<opset10::Convolution>(reshape,
                                                                   reshape_conv_weights,
                                                                   Strides(2, 1),
                                                                   CoordinateDiff(2, 0),
                                                                   CoordinateDiff(2, 0),
                                                                   Strides(2, 1));

        auto reshape_const_1 = opset10::Constant::create(element::i64, Shape{4}, {1, 5, 24, 24});
        auto reshape_1 = std::make_shared<opset10::Reshape>(reshape_conv, reshape_const_1, true);

        auto add = std::make_shared<opset10::Add>(first_conv, reshape_1);

        auto last_conv_weights = create_constant_with_zeros({8, 5, 8, 8}, {{1, 2, 3}, {}, {}, {}});
        auto last_conv = std::make_shared<opset10::Convolution>(add,
                                                                last_conv_weights,
                                                                Strides(2, 1),
                                                                CoordinateDiff(2, 0),
                                                                CoordinateDiff(2, 0),
                                                                Strides(2, 1));

        model_ref = std::make_shared<ov::Model>(OutputVector{last_conv}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeDownAdd.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(reshape_conv_weights.get_node_shared_ptr()->output(0)),
                  Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(reshape_conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(last_conv_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, MaskPropagationStopReshapeUp) {
    auto inputShapes = PartialShape{1, 6, 8, 8};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 3, 128, 1});
    auto reshape = std::make_shared<opset10::Reshape>(conv, reshape_const, true);

    auto conv_1_shape = Shape{6, 3, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(reshape,
                                                         conv_1_weights,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    auto model = std::make_shared<ov::Model>(OutputVector{conv_1}, ParameterVector{input});
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationStopReshapeUp.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)), Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);
    }
}

TEST(TransformationTests, MaskPropagationStopReshapeDown) {
    auto inputShapes = PartialShape{1, 1, 24, 24};
    auto weightsShape = Shape{8, 1, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto first_conv = std::make_shared<opset10::Convolution>(input,
                                                             weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

    auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 32, 12, 12});
    auto reshape = std::make_shared<opset10::Reshape>(first_conv, reshape_const, true);

    auto reshape_conv_weights = create_constant_with_zeros({8, 32, 13, 13}, {{1, 2, 3}, {}, {}, {}});
    auto reshape_conv = std::make_shared<opset10::Convolution>(reshape,
                                                               reshape_conv_weights,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 12),
                                                               CoordinateDiff(2, 12),
                                                               Strides(2, 1));

    auto mul = std::make_shared<opset10::Multiply>(first_conv, reshape_conv);

    auto last_conv_weights = create_constant_with_zeros({8, 8, 8, 8}, {{1, 2, 3}, {}, {}, {}});
    auto last_conv = std::make_shared<opset10::Convolution>(mul,
                                                            last_conv_weights,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

    auto model = std::make_shared<ov::Model>(OutputVector{last_conv}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationStopReshapeDown.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(reshape_conv_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(reshape_conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(last_conv_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)), Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);
    }
}

TEST_F(TransformationTestsF, MaskPropagationReshapeUnsqueezeUp) {
    auto inputShapes = PartialShape{1, 3};
    auto weightsShape = Shape{3, 12};
    auto weightsLeftUpShape = Shape{3, 6};
    auto weightsLeftShape = Shape{1, 6};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto right_weights = create_constant_with_zeros(weightsShape, {{}, {}});
    auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

    auto reshape_const = opset10::Constant::create(element::i64, {3}, {0, 6, 2});
    auto reshape = std::make_shared<opset10::Reshape>(mul_right, reshape_const, true);

    auto left_up_weights = create_constant_with_zeros(weightsLeftUpShape, {{}, {1, 2}});
    auto mul_left_up = std::make_shared<opset10::MatMul>(input, left_up_weights);

    auto mul_left = std::make_shared<opset10::MatMul>(mul_left_up, reshape);

    model = std::make_shared<ov::Model>(OutputVector{mul_left}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto right_weights = create_constant_with_zeros({weightsShape[0], weightsShape[1] - 4}, {{}, {}});
        auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

        auto reshape_const = opset10::Constant::create(element::i64, {3}, {0, 4, 2});
        auto reshape = std::make_shared<opset10::Reshape>(mul_right, reshape_const, true);

        auto left_up_weights = create_constant_with_zeros({weightsLeftUpShape[0], weightsLeftUpShape[1] - 2}, {{}, {}});
        auto mul_left_up = std::make_shared<opset10::MatMul>(input, left_up_weights);

        auto mul_left = std::make_shared<opset10::MatMul>(mul_left_up, reshape);

        model_ref = std::make_shared<ov::Model>(OutputVector{mul_left}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeUnsqueezeUp.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(right_weights.get_node_shared_ptr()->output(0)), Mask({{}, {2, 3, 4, 5}}));
    compare_masks(*getMask(mul_right->output(0)), Mask({{}, {2, 3, 4, 5}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask({{}, {1, 2}, {}}));
    compare_masks(*getMask(reshape->output(0)), Mask({{}, {1, 2}, {}}));

    compare_masks(*getMask(left_up_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2}}));
    compare_masks(*getMask(mul_left_up->output(0)), Mask({{}, {1, 2}}));
    compare_masks(*getMask(mul_left->output(0)), Mask({{}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, MaskPropagationReshapeUnsqueezeDown) {
    auto inputShapes = PartialShape{1, 3};
    auto weightsShape = Shape{3, 12};
    auto weightsLeftShape = Shape{6, 3};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    /* 1 -> 0 ch, shoudn't be pruned
       2, 3 -> 1 ch
       4, 5 -> 2 ch */
    auto right_weights = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3, 4, 5}});
    auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

    auto reshape_const = opset10::Constant::create(element::i64, {3}, {0, 6, 2});
    auto reshape = std::make_shared<opset10::Reshape>(mul_right, reshape_const, true);

    auto transpose_const = opset10::Constant::create(element::i64, {3}, {0, 2, 1});
    auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

    auto left_weights = create_constant_with_zeros(weightsLeftShape, {{}, {1, 2}});

    auto mul_left = std::make_shared<opset10::MatMul>(transpose, left_weights);

    model = std::make_shared<ov::Model>(OutputVector{mul_left}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto right_weights = create_constant_with_zeros({weightsShape[0], weightsShape[1] - 4}, {{}, {}});
        auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

        auto reshape_const = opset10::Constant::create(element::i64, {3}, {0, 4, 2});
        auto reshape = std::make_shared<opset10::Reshape>(mul_right, reshape_const, true);

        auto transpose_const = opset10::Constant::create(element::i64, {3}, {0, 2, 1});
        auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

        auto left_weights = create_constant_with_zeros({weightsLeftShape[0] - 2, weightsLeftShape[1]}, {{}, {}});

        auto mul_left = std::make_shared<opset10::MatMul>(transpose, left_weights);
        model_ref = std::make_shared<ov::Model>(OutputVector{mul_left}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeUnsqueezeDown.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(right_weights.get_node_shared_ptr()->output(0)), Mask({{}, {2, 3, 4, 5}}));
    compare_masks(*getMask(mul_right->output(0)), Mask({{}, {2, 3, 4, 5}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask({{}, {1, 2}, {}}));
    compare_masks(*getMask(reshape->output(0)), Mask({{}, {1, 2}, {}}));

    compare_masks(*getMask(transpose->output(0)), Mask({{}, {}, {1, 2}}));

    compare_masks(*getMask(left_weights.get_node_shared_ptr()->output(0)), Mask({{1, 2}, {}}));
    compare_masks(*getMask(mul_left->output(0)), Mask({{}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, MaskPropagationWrongDimsElementwise) {
    auto inputShapes = PartialShape{1, 1, 24, 24};
    auto weightsShape = Shape{8, 1, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto first_conv = std::make_shared<opset10::Convolution>(input,
                                                             weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

    auto branch_conv_weights = create_constant_with_zeros({32, 8, 2, 2}, {{1, 2, 3}, {}, {}, {}});
    auto branch_conv = std::make_shared<opset10::Convolution>(first_conv,
                                                              branch_conv_weights,
                                                              Strides(2, 2),
                                                              CoordinateDiff(2, 0),
                                                              CoordinateDiff(2, 0),
                                                              Strides(2, 1));

    auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 32, 12, 12});
    auto reshape = std::make_shared<opset10::Reshape>(first_conv, reshape_const, true);

    auto mul = std::make_shared<opset10::Multiply>(branch_conv, reshape);

    auto last_conv_weights = create_constant_with_zeros({8, 32, 8, 8}, {{1, 2, 3}, {}, {}, {}});
    auto last_conv = std::make_shared<opset10::Convolution>(mul,
                                                            last_conv_weights,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

    auto model = std::make_shared<ov::Model>(OutputVector{last_conv}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationWrongDimsElementwise.svg")
            .run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::Pruning>();
    m.run_passes(model);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(branch_conv_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(branch_conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(last_conv_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)), Mask({{}, {}, {}, {}}));
}

TEST_F(TransformationTestsF, PruneSEBlock) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto first_conv_weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto first_conv = std::make_shared<opset10::Convolution>(input,
                                                             first_conv_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));
    auto reduce_const = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
    auto reduce_mean = std::make_shared<opset10::ReduceMean>(first_conv, reduce_const, false);

    auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 6, 1, 1});
    auto reshape = std::make_shared<opset10::Reshape>(reduce_mean, reshape_const, true);

    auto se_conv_0_shape = Shape{weightsShape[0], weightsShape[0], 1, 1};
    auto se_conv_0_weights = create_constant_with_zeros(se_conv_0_shape, {{1, 2, 3}, {}, {}, {}});
    auto se_conv_0 = std::make_shared<opset10::Convolution>(reshape,
                                                            se_conv_0_weights,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
    auto se_conv_1_shape = Shape{weightsShape[0], weightsShape[0], 1, 1};
    auto se_conv_1_weights = create_constant_with_zeros(se_conv_0_shape, {{1, 2, 3}, {}, {}, {}});
    auto se_conv_1 = std::make_shared<opset10::Convolution>(se_conv_0,
                                                            se_conv_1_weights,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

    auto mul = std::make_shared<opset10::Multiply>(se_conv_1, first_conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset10::Convolution>(mul,
                                                           weights_end_conv,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    model = std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto first_conv_weights =
            create_constant_with_zeros({weightsShape[0] - 3, weightsShape[1], weightsShape[2], weightsShape[3]},
                                       {{}, {}, {}, {}});
        auto first_conv = std::make_shared<opset10::Convolution>(input,
                                                                 first_conv_weights,
                                                                 Strides(2, 1),
                                                                 CoordinateDiff(2, 0),
                                                                 CoordinateDiff(2, 0),
                                                                 Strides(2, 1));
        auto reduce_const = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce_mean = std::make_shared<opset10::ReduceMean>(first_conv, reduce_const, false);

        auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, {1, 3, 1, 1});
        auto reshape = std::make_shared<opset10::Reshape>(reduce_mean, reshape_const, true);

        auto se_conv_0_shape = Shape{weightsShape[0] - 3, weightsShape[0] - 3, 1, 1};
        auto se_conv_0_weights = create_constant_with_zeros(se_conv_0_shape, {{}, {}, {}, {}});
        auto se_conv_0 = std::make_shared<opset10::Convolution>(reshape,
                                                                se_conv_0_weights,
                                                                Strides(2, 1),
                                                                CoordinateDiff(2, 0),
                                                                CoordinateDiff(2, 0),
                                                                Strides(2, 1));
        auto se_conv_1_shape = Shape{weightsShape[0] - 3, weightsShape[0] - 3, 1, 1};
        auto se_conv_1_weights = create_constant_with_zeros(se_conv_0_shape, {{}, {}, {}, {}});
        auto se_conv_1 = std::make_shared<opset10::Convolution>(se_conv_0,
                                                                se_conv_1_weights,
                                                                Strides(2, 1),
                                                                CoordinateDiff(2, 0),
                                                                CoordinateDiff(2, 0),
                                                                Strides(2, 1));

        auto mul = std::make_shared<opset10::Multiply>(se_conv_1, first_conv);

        auto end_conv_shape = Shape{weightsShape[1], weightsShape[0] - 3, 1, 1};
        auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{}, {}, {}, {}});
        auto end_conv = std::make_shared<opset10::Convolution>(mul,
                                                               weights_end_conv,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));

        model_ref = std::make_shared<ov::Model>(OutputVector{end_conv}, ParameterVector{input}, "SEBlock");
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneSEBlock.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(first_conv_weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(se_conv_0_weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(se_conv_0->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(se_conv_1_weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(se_conv_1->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)), Mask({{}, {}, {}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksLinear) {
    const auto linear_input_features = 62 * 62 * 6;
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_linear_shape{linear_input_features, 100};
    Shape weights_last_linear_shape{100, 10};

    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{0, 1, 2}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto reshape_const = opset10::Constant::create(element::i64, Shape{2}, {1, linear_input_features});
    auto reshape = std::make_shared<opset10::Reshape>(relu, reshape_const, true);

    auto weights_linear = create_constant_with_zeros(weights_linear_shape, {{}, {0, 1, 2}});
    auto linear = std::make_shared<opset10::MatMul>(reshape, weights_linear);

    // Do net search 0 dim zeros by now
    // Check stop mask prop for outer dim (1)
    auto weights_last_linear = create_constant_with_zeros(weights_last_linear_shape, {{3, 4, 5}, {2, 3, 4}});
    auto last_linear = std::make_shared<opset10::MatMul>(linear, weights_last_linear);
    model = std::make_shared<Model>(NodeVector{last_linear}, ParameterVector{input});

    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights = create_constant_with_zeros(
            {
                weights_shape[0] - 3,
                weights_shape[1],
                weights_shape[2],
                weights_shape[3],
            },
            {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
        auto relu = std::make_shared<opset10::Relu>(conv);

        auto reshape_const = opset10::Constant::create(element::i64, Shape{2}, {1, linear_input_features / 2});
        auto reshape = std::make_shared<opset10::Reshape>(relu, reshape_const, true);

        auto weights_linear = create_constant_with_zeros(
            {
                weights_linear_shape[0] / 2,
                weights_linear_shape[1] - 3,
            },
            {{}, {}});
        auto linear = std::make_shared<opset10::MatMul>(reshape, weights_linear);

        auto weights_last_linear = create_constant_with_zeros(
            {
                weights_last_linear_shape[0] - 3,
                weights_last_linear_shape[1],
            },
            {{}, {}});
        auto last_linear = std::make_shared<opset10::MatMul>(linear, weights_last_linear);
        model_ref = std::make_shared<Model>(NodeVector{last_linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksLinear.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {0, 1, 2}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {0, 1, 2}, {}, {}}));
    auto ref_flatten_mask = std::set<uint64_t>();
    for (uint64_t i = 0; i < linear_input_features / 2; ++i)
        ref_flatten_mask.insert(i);

    using nested_vector = std::vector<std::set<uint64_t>>;
    auto reshape_ref_mask = nested_vector();
    reshape_ref_mask.push_back({});
    reshape_ref_mask.push_back(ref_flatten_mask);
    auto linear_ref_mask = nested_vector();
    linear_ref_mask.push_back(ref_flatten_mask);
    linear_ref_mask.push_back({0, 1, 2});

    compare_masks(*getMask(reshape_const->output(0)), Mask(reshape_ref_mask));
    compare_masks(*getMask(reshape->output(0)), Mask(reshape_ref_mask));
    compare_masks(*getMask(weights_linear.get_node_shared_ptr()->output(0)), Mask(linear_ref_mask));
    compare_masks(*getMask(linear->output(0)), Mask{{}, {0, 1, 2}});
    compare_masks(*getMask(weights_last_linear.get_node_shared_ptr()->output(0)), Mask{{0, 1, 2}, {}});
    compare_masks(*getMask(last_linear->output(0)), Mask{{}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, MaskPropagationMatMulStopEmptyABranch) {
    // This test rely on correct reshape mask propagation
    auto inputShapes = PartialShape{1, 3};
    auto weightsShape = Shape{3, 12};
    auto weightsLeftShape = Shape{5, 6};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto right_weights = create_constant_with_zeros(weightsShape, {{}, {0, 1, 2, 3, 4}});
    auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

    auto reshape_const = opset10::Constant::create(element::i64, {3}, {0, 6, 2});
    auto reshape = std::make_shared<opset10::Reshape>(mul_right, reshape_const, true);

    auto left_weights = create_constant_with_zeros(weightsLeftShape, {{}, {1, 2, 3, 4}});
    auto mul_left = std::make_shared<opset10::MatMul>(left_weights, reshape);

    auto model = std::make_shared<ov::Model>(OutputVector{mul_left}, ParameterVector{input});
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationMatMulStopEmptyABranch.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(right_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}}));
    compare_masks(*getMask(mul_right->output(0)), Mask({{}, {}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask({{}, {}, {}}));
    compare_masks(*getMask(reshape->output(0)), Mask({{}, {}, {}}));

    check_mask_is_not_exist(getMask(left_weights.get_node_shared_ptr()->output(0)));
    compare_masks(*getMask(mul_left->output(0)), Mask({{}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);
    }
}

TEST(TransformationTests, PruneLinearUp) {
    const auto linear_input_features = 6 * 2 * 2;
    auto inputShapes = PartialShape{1, 6, 2, 2};
    auto weightsShape = Shape{6, 6, 1, 1};
    auto linearShape = Shape{linear_input_features, linear_input_features};
    auto lastLinearShape = Shape{10, linear_input_features};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto reshape_const = opset10::Constant::create(element::i64, Shape{2}, {1, linear_input_features});
    auto reshape = std::make_shared<opset10::Reshape>(conv, reshape_const, true);

    auto linear_mask = Mask();
    auto outer_dim_zeros = std::set<uint64_t>();
    for (auto i = 0; i < linear_input_features / 2; ++i)
        outer_dim_zeros.insert(i);
    linear_mask.push_back({10, 11});
    linear_mask.push_back(outer_dim_zeros);
    auto linear_const = create_constant_with_zeros(linearShape, linear_mask);
    auto linear = std::make_shared<opset10::MatMul>(reshape, linear_const);

    auto add_mask = Mask();
    add_mask.push_back({});
    add_mask.push_back(outer_dim_zeros);
    auto add_const = create_constant_with_zeros({1, linear_input_features}, add_mask);
    auto add = std::make_shared<opset10::Add>(linear, add_const);
    auto add_const_1 = create_constant_with_zeros({1, linear_input_features}, add_mask);
    auto add_1 = std::make_shared<opset10::Add>(add, add_const_1);
    auto add_2 = std::make_shared<opset10::Add>(add_1, reshape);

    auto bad_add_const = create_constant_with_zeros({1, linear_input_features}, {{}, {}});
    auto bad_add = std::make_shared<opset10::Add>(add_2, bad_add_const);

    auto weights_end_linear = create_constant_with_zeros(lastLinearShape, {{1, 2, 3}, {3, 4, 6}});
    auto last_linear = std::make_shared<opset10::MatMul>(bad_add, weights_end_linear, false, true);
    auto model = std::make_shared<ov::Model>(OutputVector{last_linear}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneLinearUp.svg").run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::Pruning>();
    m.run_passes(model);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_linear.get_node_shared_ptr()->output(0)), Mask({{}, {}}));
    compare_masks(*getMask(last_linear->output(0)), Mask({{}, {}}));
}

TEST(TransformationTests, PruneConvUpShort) {
    const auto linear_input_features = 6 * 2 * 2;
    auto inputShapes = PartialShape{1, 6, 2, 2};
    auto convShape = Shape{1, 6, 2, 2};
    auto weightsShape = Shape{6, 6, 1, 1};
    auto lastLinearShape = Shape{10, linear_input_features};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));

    auto conv_1_const = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset10::Convolution>(conv,
                                                         conv_1_const,
                                                         Strides(2, 1),
                                                         CoordinateDiff(2, 0),
                                                         CoordinateDiff(2, 0),
                                                         Strides(2, 1));

    auto add_const = create_constant_with_zeros(convShape, {{}, {1, 2, 3}, {}, {}});
    auto add = std::make_shared<opset10::Add>(conv_1, add_const);
    auto add_const_1 = create_constant_with_zeros(convShape, {{}, {1, 2, 3}, {}, {}});
    auto add_1 = std::make_shared<opset10::Add>(add, add_const_1);
    auto add_2 = std::make_shared<opset10::Add>(add_1, conv);

    auto bad_add_const = create_constant_with_zeros(convShape, {{}, {}, {}, {}});
    auto bad_add = std::make_shared<opset10::Add>(add_2, bad_add_const);

    auto weights_end_conv = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {1, 2, 3}, {}, {}});
    auto last_conv = std::make_shared<opset10::Convolution>(bad_add,
                                                            weights_end_conv,
                                                            Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

    auto model = std::make_shared<ov::Model>(OutputVector{last_conv}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneConvUpShort.svg").run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::Pruning>();
    m.run_passes(model);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)), Mask({{}, {}, {}, {}}));
}

TEST_F(TransformationTestsF, MaskPropagationLinearOuterDims) {
    auto inputShapes = PartialShape{1, 6, 3};
    auto weightsShape = Shape{3, 12};
    auto EltwiseShape = Shape{1, 6, 12};
    auto weightsLeftShape = Shape{6, 3};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    /* 1 -> 0 ch, shoudn't be pruned
       2, 3 -> 1 ch
       4, 5 -> 2 ch */
    auto right_weights = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3, 4, 5}});
    auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

    auto eltwise_mul_const = create_constant_with_zeros(EltwiseShape, {{}, {1}, {}});
    auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_const);

    auto reshape_const = opset10::Constant::create(element::i64, {4}, {0, 0, 6, 2});
    auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

    auto transpose_const = opset10::Constant::create(element::i64, {4}, {0, 2, 3, 1});
    auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

    auto left_weights = create_constant_with_zeros(weightsLeftShape, {{}, {1, 2}});

    auto mul_left = std::make_shared<opset10::MatMul>(transpose, left_weights);

    auto flatten_const = opset10::Constant::create(element::i64, {2}, {1, 36});
    auto flatten = std::make_shared<opset10::Reshape>(mul_left, flatten_const, true);

    auto last_mul_const = create_constant_with_zeros({36, 2}, {{}, {0}});
    auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

    model = std::make_shared<ov::Model>(OutputVector{last_mul}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto right_weights = create_constant_with_zeros({weightsShape[0], weightsShape[1] - 4}, {{}, {}});
        auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

        auto eltwise_mul_const =
            create_constant_with_zeros({EltwiseShape[0], EltwiseShape[1], EltwiseShape[2] - 4}, {{}, {1}, {}});
        auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_const);

        auto reshape_const = opset10::Constant::create(element::i64, {4}, {0, 0, 4, 2});
        auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

        auto transpose_const = opset10::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

        auto left_weights = create_constant_with_zeros({weightsLeftShape[0], weightsLeftShape[1] - 2}, {{}, {}});

        auto mul_left = std::make_shared<opset10::MatMul>(transpose, left_weights);

        auto flatten_const = opset10::Constant::create(element::i64, {2}, {1, 8});
        auto flatten = std::make_shared<opset10::Reshape>(mul_left, flatten_const, true);

        auto last_mul_const = create_constant_with_zeros({8, 2}, {{}, {0}});
        auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

        model_ref = std::make_shared<ov::Model>(OutputVector{last_mul}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationLinearOuterDims.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(right_weights.get_node_shared_ptr()->output(0)), Mask({{}, {2, 3, 4, 5}}));
    compare_masks(*getMask(mul_right->output(0)), Mask({{}, {}, {2, 3, 4, 5}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask({{}, {}, {1, 2}, {}}));
    compare_masks(*getMask(reshape->output(0)), Mask({{}, {}, {1, 2}, {}}));

    compare_masks(*getMask(transpose->output(0)), Mask({{}, {1, 2}, {}, {}}));

    compare_masks(*getMask(left_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2}}));
    compare_masks(*getMask(mul_left->output(0)), Mask({{}, {1, 2}, {}, {1, 2}}));

    auto ref_dim = std::set<uint64_t>();
    for (uint64_t i = 6; i < 18; ++i)
        ref_dim.insert(i);
    for (uint64_t i = 0; i < 6; ++i)
        for (auto& zero_dim : {1, 2, 4, 5})
            ref_dim.insert(i * 6 + zero_dim);

    auto ref_flatten_mask = Mask();
    ref_flatten_mask.push_back({});
    ref_flatten_mask.push_back(ref_dim);

    compare_masks(*getMask(flatten_const->output(0)), ref_flatten_mask);
    compare_masks(*getMask(flatten->output(0)), ref_flatten_mask);

    auto ref_last_mul_mask = Mask();
    ref_last_mul_mask.push_back(ref_dim);
    ref_last_mul_mask.push_back({});

    compare_masks(*getMask(last_mul_const.get_node_shared_ptr()->output(0)), ref_last_mul_mask);
    compare_masks(*getMask(last_mul->output(0)), Mask({{}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, MaskPropagationStopLinearOuterDims) {
    auto inputShapes = PartialShape{6, 3};
    auto weightsShape = Shape{3, 12};
    auto EltwiseShape = Shape{6, 12};
    auto weightsLeftShape = Shape{6, 3};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    /* 1 -> 0 ch, shoudn't be pruned
       2, 3 -> 1 ch
       4, 5 -> 2 ch */
    auto right_weights = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3, 4, 5}});
    auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

    auto eltwise_mul_const = create_constant_with_zeros(EltwiseShape, {{1}, {}});
    auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_const);

    auto reshape_const = opset10::Constant::create(element::i64, {3}, {0, 6, 2});
    auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

    auto transpose_const = opset10::Constant::create(element::i64, {3}, {1, 2, 0});
    auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

    auto left_weights = create_constant_with_zeros(weightsLeftShape, {{}, {1, 2}});

    auto mul_left = std::make_shared<opset10::MatMul>(transpose, left_weights);

    auto flatten_const = opset10::Constant::create(element::i64, {2}, {1, 36});
    auto flatten = std::make_shared<opset10::Reshape>(mul_left, flatten_const, true);

    auto last_mul_const = create_constant_with_zeros({36, 2}, {{}, {0}});
    auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

    auto model = std::make_shared<ov::Model>(OutputVector{last_mul}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationStopLinearOuterDims.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(right_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}}));
    compare_masks(*getMask(mul_right->output(0)), Mask({{}, {}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask({{}, {}, {}}));
    compare_masks(*getMask(reshape->output(0)), Mask({{}, {}, {}}));

    compare_masks(*getMask(transpose->output(0)), Mask({{}, {}, {}}));

    compare_masks(*getMask(left_weights.get_node_shared_ptr()->output(0)), Mask({{}, {}}));
    compare_masks(*getMask(mul_left->output(0)), Mask({{}, {}, {}}));

    compare_masks(*getMask(flatten_const->output(0)), Mask({{}, {}}));
    compare_masks(*getMask(flatten->output(0)), Mask({{}, {}}));

    compare_masks(*getMask(last_mul_const.get_node_shared_ptr()->output(0)), Mask({{}, {}}));
    compare_masks(*getMask(last_mul->output(0)), Mask({{}, {}}));
    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);
    }
}

TEST_F(TransformationTestsF, PruneMasksMatMulColsStopRowsUp) {
    const auto linear_input_features = 62 * 62;
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_linear_shape{linear_input_features, 100};
    Shape weights_last_linear_shape{100, 10};

    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{0, 1, 2}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto reshape_const = opset10::Constant::create(element::i64, Shape{3}, {1, 6, linear_input_features});
    auto reshape = std::make_shared<opset10::Reshape>(relu, reshape_const, true);

    auto weights_linear = create_constant_with_zeros(weights_linear_shape, {{}, {0, 1, 2}});
    auto linear = std::make_shared<opset10::MatMul>(reshape, weights_linear);

    // Do net search 0 dim zeros by now
    auto weights_last_linear = create_constant_with_zeros(weights_last_linear_shape, {{3, 4, 5}, {}});
    auto last_linear = std::make_shared<opset10::MatMul>(linear, weights_last_linear);
    model = std::make_shared<Model>(NodeVector{last_linear}, ParameterVector{input});

    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights = create_constant_with_zeros(weights_shape, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
        auto relu = std::make_shared<opset10::Relu>(conv);

        auto reshape_const = opset10::Constant::create(element::i64, Shape{3}, {1, 6, linear_input_features});
        auto reshape = std::make_shared<opset10::Reshape>(relu, reshape_const, true);

        auto weights_linear = create_constant_with_zeros(
            {
                weights_linear_shape[0],
                weights_linear_shape[1] - 3,
            },
            {{}, {}});
        auto linear = std::make_shared<opset10::MatMul>(reshape, weights_linear);

        auto weights_last_linear = create_constant_with_zeros(
            {
                weights_last_linear_shape[0] - 3,
                weights_last_linear_shape[1],
            },
            {{}, {}});
        auto last_linear = std::make_shared<opset10::MatMul>(linear, weights_last_linear);
        model_ref = std::make_shared<Model>(NodeVector{last_linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneMasksMatMulColsStopRowsUp.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask{{}, {}, {}});
    compare_masks(*getMask(reshape->output(0)), Mask{{}, {}, {}});
    compare_masks(*getMask(weights_linear.get_node_shared_ptr()->output(0)), Mask({{}, {0, 1, 2}}));
    compare_masks(*getMask(linear->output(0)), Mask{{}, {}, {0, 1, 2}});
    compare_masks(*getMask(weights_last_linear.get_node_shared_ptr()->output(0)), Mask{{0, 1, 2}, {}});
    compare_masks(*getMask(last_linear->output(0)), Mask{{}, {}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PruneMasksMatMulRowsStopColsUp) {
    // Checks rows matmul pruning + transpose input in matmul
    const auto linear_input_features = 62 * 62;
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_linear_shape{linear_input_features, 100};
    Shape weights_last_linear_shape{10, 6};

    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{0, 1, 2}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto reshape_const = opset10::Constant::create(element::i64, Shape{3}, {1, 6, linear_input_features});
    auto reshape = std::make_shared<opset10::Reshape>(relu, reshape_const, true);

    auto weights_linear = create_constant_with_zeros(weights_linear_shape, {{3, 4, 5}, {3, 4}});
    auto linear = std::make_shared<opset10::MatMul>(reshape, weights_linear);

    // Do net search this zeros by now
    auto weights_last_linear = create_constant_with_zeros(weights_last_linear_shape, {{}, {3, 4, 5}});
    // To prune rows we should transpose featuremap. Did it by transpose_a = true MatMul constructor attr
    auto last_linear = std::make_shared<opset10::MatMul>(linear, weights_last_linear, true, true);
    model = std::make_shared<Model>(NodeVector{last_linear}, ParameterVector{input});

    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights = create_constant_with_zeros(
            {
                weights_shape[0] - 3,
                weights_shape[1],
                weights_shape[2],
                weights_shape[3],
            },
            {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
        auto relu = std::make_shared<opset10::Relu>(conv);

        auto reshape_const = opset10::Constant::create(element::i64, Shape{3}, {1, 3, linear_input_features});
        auto reshape = std::make_shared<opset10::Reshape>(relu, reshape_const, true);

        auto weights_linear = create_constant_with_zeros(
            {
                weights_linear_shape[0],
                weights_linear_shape[1],
            },
            {{}, {}});
        auto linear = std::make_shared<opset10::MatMul>(reshape, weights_linear);

        auto weights_last_linear =
            create_constant_with_zeros({weights_last_linear_shape[0], weights_last_linear_shape[1] - 3}, {{}, {}});
        // To prune rows we should transpose featuremap. Did it by transpose_a = true MatMul constructor attr
        auto last_linear = std::make_shared<opset10::MatMul>(linear, weights_last_linear, true, true);
        model_ref = std::make_shared<Model>(NodeVector{last_linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneMasksMatMulRowsStopColsUp.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {0, 1, 2}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {0, 1, 2}, {}, {}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask{{}, {0, 1, 2}, {}});
    compare_masks(*getMask(reshape->output(0)), Mask{{}, {0, 1, 2}, {}});
    compare_masks(*getMask(weights_linear.get_node_shared_ptr()->output(0)), Mask{{}, {}});
    compare_masks(*getMask(linear->output(0)), Mask{{}, {0, 1, 2}, {}});
    compare_masks(*getMask(weights_last_linear.get_node_shared_ptr()->output(0)), Mask{{}, {0, 1, 2}});
    compare_masks(*getMask(last_linear->output(0)), Mask{{}, {}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateFlattenUp) {
    // Propagate Flatten down is the same as in
    // TODO: Make this test
    // PruneLinearIsClosingAndInGroup test
    using nested_vector = std::vector<std::set<uint64_t>>;
    constexpr auto linear_input_features = 6 * 8 * 8;
    Shape input_shape{1, 3, 8, 8};
    Shape weights_shape{6, 3, 1, 1};
    Shape weights_linear_shape{linear_input_features, 100};

    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{0, 1, 2}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto relu = std::make_shared<opset10::Relu>(conv);

    auto reshape_const = opset10::Constant::create(element::i64, Shape{2}, {1, linear_input_features});
    auto reshape = std::make_shared<opset10::Reshape>(relu, reshape_const, true);

    // Skip just one zero in dim should lead to
    // whole dimension invalidating.
    auto add_zeros = std::set<uint64_t>();
    for (size_t i = 1; i < linear_input_features / 2; i++)
        add_zeros.insert(i);
    auto add_mask = nested_vector();
    add_mask.push_back({});
    add_mask.push_back(add_zeros);
    auto weights_add = create_constant_with_zeros({1, linear_input_features}, Mask(add_mask));
    auto add = std::make_shared<opset10::Add>(reshape, weights_add);

    auto weights_linear = create_constant_with_zeros(weights_linear_shape, {{}, {0, 1, 2}});
    auto linear = std::make_shared<opset10::MatMul>(add, weights_linear);

    model = std::make_shared<Model>(NodeVector{linear}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);
        auto weights = create_constant_with_zeros(
            {
                weights_shape[0] - 2,
                weights_shape[1],
                weights_shape[2],
                weights_shape[3],
            },
            {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
        auto relu = std::make_shared<opset10::Relu>(conv);

        auto reshape_const = opset10::Constant::create(element::i64, Shape{2}, {1, 2 * linear_input_features / 3});
        auto reshape = std::make_shared<opset10::Reshape>(relu, reshape_const, true);

        auto weights_add = create_constant_with_zeros({1, 2 * linear_input_features / 3}, Mask{{}, {}});
        auto add = std::make_shared<opset10::Add>(reshape, weights_add);

        auto weights_linear = create_constant_with_zeros(
            {
                2 * weights_linear_shape[0] / 3,
                weights_linear_shape[1],
            },
            {{}, {}});
        auto linear = std::make_shared<opset10::MatMul>(add, weights_linear);

        model_ref = std::make_shared<Model>(NodeVector{linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateFlattenUp.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {1, 2}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {1, 2}, {}, {}}));
    auto ref_flatten_mask = std::set<uint64_t>();
    for (uint64_t i = linear_input_features / 6; i < linear_input_features / 2; ++i)
        ref_flatten_mask.insert(i);

    auto reshape_ref_mask = nested_vector();
    reshape_ref_mask.push_back({});
    reshape_ref_mask.push_back(ref_flatten_mask);
    auto linear_ref_mask = nested_vector();
    linear_ref_mask.push_back(ref_flatten_mask);
    linear_ref_mask.push_back({});

    compare_masks(*getMask(reshape_const->output(0)), Mask(reshape_ref_mask));
    compare_masks(*getMask(reshape->output(0)), Mask(reshape_ref_mask));
    compare_masks(*getMask(weights_linear.get_node_shared_ptr()->output(0)), Mask(linear_ref_mask));
    compare_masks(*getMask(linear->output(0)), Mask{{}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateFlattenDown) {
    const auto linear_input_features = 6 * 2 * 2;
    auto inputShapes = PartialShape{1, 6, 2, 2};
    auto weightsShape = Shape{6, 6, 1, 1};
    auto linearShape = Shape{linear_input_features, linear_input_features};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset10::Convolution>(input,
                                                       weights,
                                                       Strides(2, 1),
                                                       CoordinateDiff(2, 0),
                                                       CoordinateDiff(2, 0),
                                                       Strides(2, 1));
    auto reshape_const = opset10::Constant::create(element::i64, Shape{2}, {1, linear_input_features});
    auto reshape = std::make_shared<opset10::Reshape>(conv, reshape_const, true);
    auto linear_mask = Mask();
    auto outer_dim_zeros = std::set<uint64_t>();
    for (auto i = 0; i < linear_input_features / 2; ++i)
        outer_dim_zeros.insert(i);
    linear_mask.push_back({10, 11});
    linear_mask.push_back(outer_dim_zeros);
    auto linear_const = create_constant_with_zeros(linearShape, linear_mask);
    auto linear = std::make_shared<opset10::MatMul>(reshape, linear_const);
    model = std::make_shared<ov::Model>(OutputVector{linear}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto weights =
            create_constant_with_zeros({weightsShape[0] - 2, weightsShape[1], weightsShape[2], weightsShape[3]},
                                       {{}, {}, {}, {}});
        auto conv = std::make_shared<opset10::Convolution>(input,
                                                           weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
        auto reshape_const = opset10::Constant::create(element::i64, Shape{2}, {1, 2 * linear_input_features / 3});
        auto reshape = std::make_shared<opset10::Reshape>(conv, reshape_const, true);
        auto linear_const =
            create_constant_with_zeros({linearShape[0] - linear_input_features / 3, linearShape[1]}, {{}, {}});
        auto linear = std::make_shared<opset10::MatMul>(reshape, linear_const);
        model_ref = std::make_shared<ov::Model>(OutputVector{linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateFlattenDown.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    auto weights_mask = getMask(weights.get_node_shared_ptr()->output(0));
    weights_mask->at(0) = std::set<uint64_t>({2, 3});
    auto conv_mask = getMask(conv->output(0));
    conv_mask->apply_callback(weights_mask);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {2, 3}, {}, {}}));
    outer_dim_zeros.clear();
    for (auto i = 2 * linear_input_features / 6; i < 2 * linear_input_features / 3; ++i)
        outer_dim_zeros.insert(i);

    linear_mask[0] = outer_dim_zeros;
    linear_mask[1] = {};
    compare_masks(*getMask(linear_const.get_node_shared_ptr()->output(0)), linear_mask);
    compare_masks(*getMask(linear->output(0)), {{}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksTranspose) {
    Shape input_shape{1, 8};
    Shape weights_shape{5, 8};

    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);

    auto weights = create_constant_with_zeros(weights_shape, {{1, 2, 3}, {}});
    auto transpose_const = opset10::Constant::create(element::i64, {2}, {1, 0});
    auto transpose = std::make_shared<opset10::Transpose>(weights, transpose_const);
    auto mul = std::make_shared<opset10::MatMul>(input, transpose);
    auto relu = std::make_shared<opset10::Relu>(mul);

    auto last_mul_weights = create_constant_with_zeros(weights_shape, {{}, {1, 2, 3}});
    auto last_mul = std::make_shared<opset10::MatMul>(relu, last_mul_weights);

    model = std::make_shared<Model>(NodeVector{last_mul}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);

        auto weights = create_constant_with_zeros({weights_shape[0] - 3, weights_shape[1]}, {{}, {}});
        auto transpose_const = opset10::Constant::create(element::i64, {2}, {1, 0});
        auto transpose = std::make_shared<opset10::Transpose>(weights, transpose_const);
        auto mul = std::make_shared<opset10::MatMul>(input, transpose);
        auto relu = std::make_shared<opset10::Relu>(mul);

        auto last_mul_weights = create_constant_with_zeros({weights_shape[0] - 3, weights_shape[1]}, {{}, {}});
        auto last_mul = std::make_shared<opset10::MatMul>(relu, last_mul_weights);

        model_ref = std::make_shared<Model>(NodeVector{last_mul}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksTranspose.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{1, 2, 3}, {}}));
    compare_masks(*getMask(transpose->output(0)), Mask({{}, {1, 2, 3}}));
    compare_masks(*getMask(mul->output(0)), Mask({{}, {1, 2, 3}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {1, 2, 3}}));
    compare_masks(*getMask(last_mul->output(0)), Mask{{}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksTransposeComplex) {
    Shape input_shape{1, 3, 5, 7, 8};
    Shape weights_shape{5, 7, 3, 8, 1};
    Shape last_mul_weights_shape{7, 5};

    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);

    auto weights = create_constant_with_zeros(weights_shape, {{}, {1, 2, 3}, {}, {}, {}});

    const auto transpose_consts =
        std::vector<std::vector<int64_t>>{{4, 0, 3, 1, 2}, {0, 4, 1, 3, 2}, {4, 3, 2, 1, 0}, {4, 3, 2, 0, 1}};
    auto last_output = weights;
    for (auto& transpose_const_vec : transpose_consts) {
        auto transpose_const =
            opset10::Constant::create(element::i64, {transpose_const_vec.size()}, transpose_const_vec);
        last_output = std::make_shared<opset10::Transpose>(last_output, transpose_const);
    }

    auto mul = std::make_shared<opset10::MatMul>(input, last_output);
    auto relu = std::make_shared<opset10::Relu>(mul);

    auto last_mul_weights = create_constant_with_zeros(last_mul_weights_shape, {{}, {1, 2, 3}});
    auto last_mul = std::make_shared<opset10::MatMul>(relu, last_mul_weights);

    model = std::make_shared<Model>(NodeVector{last_mul}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);

        auto weights = create_constant_with_zeros(
            {
                weights_shape[0],
                weights_shape[1] - 3,
                weights_shape[2],
                weights_shape[3],
                weights_shape[4],
            },
            {{}, {}});
        const auto transpose_consts =
            std::vector<std::vector<int64_t>>{{4, 0, 3, 1, 2}, {0, 4, 1, 3, 2}, {4, 3, 2, 1, 0}, {4, 3, 2, 0, 1}};
        auto last_output = weights;
        for (auto& transpose_const_vec : transpose_consts) {
            auto transpose_const =
                opset10::Constant::create(element::i64, {transpose_const_vec.size()}, transpose_const_vec);
            last_output = std::make_shared<opset10::Transpose>(last_output, transpose_const);
        }
        auto mul = std::make_shared<opset10::MatMul>(input, last_output);
        auto relu = std::make_shared<opset10::Relu>(mul);

        auto last_mul_weights =
            create_constant_with_zeros({last_mul_weights_shape[0] - 3, last_mul_weights_shape[1]}, {{}, {}});
        auto last_mul = std::make_shared<opset10::MatMul>(relu, last_mul_weights);

        model_ref = std::make_shared<Model>(NodeVector{last_mul}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksTransposeComplex.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{}, {}, {}, {}, {1, 2, 3}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {}, {}, {}, {1, 2, 3}}));
    compare_masks(*getMask(last_mul_weights), Mask{{1, 2, 3}, {}});
    compare_masks(*getMask(last_mul->output(0)), Mask{{}, {}, {}, {}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, PropagateMasksTransposeStop) {
    Shape input_shape{1, 8};
    Shape weights_shape{5, 8, 2, 3};
    Shape last_mul_shape{2, 3, 5, 8};

    auto input = std::make_shared<opset10::Parameter>(element::f32, input_shape);

    auto weights = create_constant_with_zeros(weights_shape, {{1, 2, 3}, {}, {}, {}});
    auto transpose_const = opset10::Constant::create(element::i64, {4}, {2, 3, 1, 0});
    auto transpose = std::make_shared<opset10::Transpose>(weights, transpose_const);

    auto shape_of = std::make_shared<opset10::ShapeOf>(transpose);
    auto reshape = std::make_shared<opset10::Reshape>(transpose, shape_of, true);

    auto mul = std::make_shared<opset10::MatMul>(input, reshape);
    auto relu = std::make_shared<opset10::Relu>(mul);

    auto last_mul_weights = create_constant_with_zeros(last_mul_shape, {{}, {}, {}, {1, 2, 3}});
    auto last_mul = std::make_shared<opset10::MatMul>(relu, last_mul_weights);

    auto model = std::make_shared<Model>(NodeVector{last_mul}, ParameterVector{input});
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksTransposeStop.svg").run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    check_mask_is_not_exist(getMask(weights.get_node_shared_ptr()->output(0)));
    check_mask_is_not_exist(getMask(transpose->output(0)));
    check_mask_is_not_exist(getMask(mul->output(0)));
    check_mask_is_not_exist(getMask(relu->output(0)));
    compare_masks(*getMask(last_mul->output(0)), Mask{{}, {}, {}, {}});
    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);
    }
}

// Reason: a net, when broadcasting is changing output values
TEST_F(DISABLED_TransformationTestsF, PropagateMasksBroadcastedEltwiseWithInputs) {
    constexpr size_t heads(6), dim_in_head(3), dim(12);
    auto inputShapes = PartialShape{2, heads, dim_in_head};
    auto weightsShape = Shape{dim_in_head, dim};
    auto EltwiseShape = Shape{2, heads, dim};
    auto broadcastedEltwiseShape = Shape{1, 1, 1, heads};
    auto broadcastedEltwiseShapeShort = Shape{2, 2, heads};
    auto weightsLeftShape = Shape{6, 3};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);

    auto right_weights = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3, 4, 5}});
    auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

    auto eltwise_mul_const = create_constant_with_zeros(EltwiseShape, {{}, {1}, {}});
    auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_const);

    auto reshape_const = opset10::Constant::create(element::i64, {4}, {0, 0, 6, 2});
    auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

    auto transpose_const = opset10::Constant::create(element::i64, {4}, {0, 2, 3, 1});
    auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

    auto broadcasted_eltwise_weights = create_constant_with_zeros(broadcastedEltwiseShape, {{0}, {}, {}, {}});
    auto broadcasted_eltwise = std::make_shared<opset10::Add>(transpose, broadcasted_eltwise_weights);
    broadcasted_eltwise->set_friendly_name("Eltwise broadcasted same rank");

    auto transpose_to_const = opset10::Constant::create(element::i64, {4}, {1, 0, 2, 3});
    auto transpose_to = std::make_shared<opset10::Transpose>(broadcasted_eltwise, transpose_to_const);

    auto broadcasted_eltwise_weights_short = create_constant_with_zeros(broadcastedEltwiseShapeShort, {{}, {0, 1}, {}});
    auto broadcasted_eltwise_short = std::make_shared<opset10::Add>(transpose_to, broadcasted_eltwise_weights_short);
    broadcasted_eltwise_short->set_friendly_name("Eltwise broadcasted smaller rank");

    auto transpose_from = std::make_shared<opset10::Transpose>(broadcasted_eltwise_short, transpose_to_const);

    auto dummy_eltwise_weights = std::make_shared<opset10::Parameter>(element::f32, broadcastedEltwiseShape);
    auto dummy_eltwise_inputs = std::make_shared<opset10::Parameter>(element::f32, broadcastedEltwiseShape);
    auto dummy_eltwise = std::make_shared<opset10::Add>(dummy_eltwise_inputs, dummy_eltwise_weights);

    auto broadcasted_eltwise_without_weights_mask = std::make_shared<opset10::Add>(transpose_from, dummy_eltwise);
    broadcasted_eltwise_without_weights_mask->set_friendly_name("Eltwise broadcasted same rank without weights mask");

    auto left_weights = create_constant_with_zeros(weightsLeftShape, {{}, {1, 2}});

    auto mul_left = std::make_shared<opset10::MatMul>(broadcasted_eltwise_without_weights_mask, left_weights);

    auto flatten_const = opset10::Constant::create(element::i64, {2}, {2, 36});
    auto flatten = std::make_shared<opset10::Reshape>(mul_left, flatten_const, true);

    auto last_mul_const = create_constant_with_zeros({36, 2}, {{}, {0}});
    auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

    model = std::make_shared<ov::Model>(OutputVector{last_mul},
                                        ParameterVector{input, dummy_eltwise_inputs, dummy_eltwise_weights});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);

        auto right_weights = create_constant_with_zeros({weightsShape[0], weightsShape[1] - 4}, {{}, {1}});
        auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

        auto eltwise_mul_const =
            create_constant_with_zeros({EltwiseShape[0], EltwiseShape[1], EltwiseShape[2] - 4}, {{}, {1}, {}});
        auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_const);

        auto reshape_const = opset10::Constant::create(element::i64, {4}, {0, 0, 4, 2});
        auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

        auto transpose_const = opset10::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

        auto broadcasted_eltwise_weights = create_constant_with_zeros(broadcastedEltwiseShape, {{0}, {}, {}, {}});
        auto broadcasted_eltwise = std::make_shared<opset10::Add>(transpose, broadcasted_eltwise_weights);

        auto transpose_to_const = opset10::Constant::create(element::i64, {4}, {1, 0, 2, 3});
        auto transpose_to = std::make_shared<opset10::Transpose>(broadcasted_eltwise, transpose_to_const);

        auto broadcasted_eltwise_weights_short =
            create_constant_with_zeros(broadcastedEltwiseShapeShort, {{}, {0, 1}, {}});
        auto broadcasted_eltwise_short =
            std::make_shared<opset10::Add>(transpose_to, broadcasted_eltwise_weights_short);

        auto transpose_from = std::make_shared<opset10::Transpose>(broadcasted_eltwise_short, transpose_to_const);

        auto dummy_eltwise_weights = std::make_shared<opset10::Parameter>(element::f32, broadcastedEltwiseShape);
        auto dummy_eltwise_inputs = std::make_shared<opset10::Parameter>(element::f32, broadcastedEltwiseShape);
        auto dummy_eltwise = std::make_shared<opset10::Add>(dummy_eltwise_inputs, dummy_eltwise_weights);

        auto broadcasted_eltwise_without_weights_mask = std::make_shared<opset10::Add>(transpose_from, dummy_eltwise);

        auto left_weights = create_constant_with_zeros({weightsLeftShape[0], weightsLeftShape[1] - 2}, {{}, {}});

        auto mul_left = std::make_shared<opset10::MatMul>(broadcasted_eltwise_without_weights_mask, left_weights);

        auto flatten_const = opset10::Constant::create(element::i64, {2}, {2, 8});
        auto flatten = std::make_shared<opset10::Reshape>(mul_left, flatten_const, true);

        auto last_mul_const = create_constant_with_zeros({8, 2}, {{}, {0}});
        auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

        model_ref = std::make_shared<ov::Model>(OutputVector{last_mul},
                                                ParameterVector{input, dummy_eltwise_inputs, dummy_eltwise_weights});
    }
    if (VISUALIZE_TESTS_TREE) {
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksBroadcastedEltwiseWithInputs.svg")
            .run_on_model(model);
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(right_weights.get_node_shared_ptr()->output(0)), Mask({{}, {2, 3, 4, 5}}));
    compare_masks(*getMask(mul_right->output(0)), Mask({{}, {}, {2, 3, 4, 5}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask({{}, {}, {1, 2}, {}}));
    compare_masks(*getMask(reshape->output(0)), Mask({{}, {}, {1, 2}, {}}));

    compare_masks(*getMask(transpose->output(0)), Mask({{}, {1, 2}, {}, {}}));
    compare_masks(*getMask(broadcasted_eltwise->output(0)), Mask({{}, {1, 2}, {}, {}}));

    compare_masks(*getMask(transpose_to), Mask{{1, 2}, {}, {}, {}});
    compare_masks(*getMask(broadcasted_eltwise_weights_short), Mask{{}, {}, {}});
    compare_masks(*getMask(broadcasted_eltwise_short->output(0)), Mask{{1, 2}, {}, {}, {}});
    compare_masks(*getMask(transpose_from), Mask{{}, {1, 2}, {}, {}});

    compare_masks(*getMask(broadcasted_eltwise_without_weights_mask->output(0)), Mask({{}, {1, 2}, {}, {}}));

    compare_masks(*getMask(left_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2}}));
    compare_masks(*getMask(mul_left->output(0)), Mask({{}, {1, 2}, {}, {1, 2}}));

    auto ref_flatten_mask = Mask();
    auto ref_dim = std::set<uint64_t>();
    for (uint64_t i = 6; i < 18; ++i)
        ref_dim.insert(i);
    for (uint64_t i = 0; i < 6; ++i)
        for (auto& zero_dim : {1, 2, 4, 5})
            ref_dim.insert(i * 6 + zero_dim);

    ref_flatten_mask.push_back({});
    ref_flatten_mask.push_back(ref_dim);

    compare_masks(*getMask(flatten_const->output(0)), ref_flatten_mask);
    compare_masks(*getMask(flatten->output(0)), ref_flatten_mask);

    auto ref_last_mul_mask = Mask();
    ref_last_mul_mask.push_back(ref_dim);
    ref_last_mul_mask.push_back({});

    compare_masks(*getMask(last_mul_const.get_node_shared_ptr()->output(0)), ref_last_mul_mask);
    compare_masks(*getMask(last_mul->output(0)), Mask({{}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PropagateMasksBroadcastedEltwise) {
    constexpr size_t heads(6), dim_in_head(3), dim(12);
    auto inputShapes = PartialShape{2, heads, dim_in_head};
    auto weightsShape = Shape{dim_in_head, dim};
    auto EltwiseShape = Shape{2, heads, dim};
    auto broadcastedEltwiseShape = Shape{1, 1, 1, heads};
    auto broadcastedEltwiseShapeShort = Shape{2, 2, heads};
    auto weightsLeftShape = Shape{6, 3};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);

    auto right_weights = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3, 4, 5}});
    auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

    auto eltwise_mul_const = create_constant_with_zeros(EltwiseShape, {{}, {1}, {}});
    auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_const);

    auto reshape_const = opset10::Constant::create(element::i64, {4}, {0, 0, 6, 2});
    auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

    auto transpose_const = opset10::Constant::create(element::i64, {4}, {0, 2, 3, 1});
    auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

    auto broadcasted_eltwise_weights = create_constant_with_zeros(broadcastedEltwiseShape, {{0}, {}, {}, {}});
    auto broadcasted_eltwise = std::make_shared<opset10::Add>(transpose, broadcasted_eltwise_weights);
    broadcasted_eltwise->set_friendly_name("Eltwise broadcasted same rank");

    auto broadcasted_eltwise2 = std::make_shared<opset10::Add>(broadcasted_eltwise_weights, broadcasted_eltwise);
    broadcasted_eltwise2->set_friendly_name("Eltwise broadcasted same rank swapped");

    auto transpose_to_const = opset10::Constant::create(element::i64, {4}, {1, 0, 2, 3});
    auto transpose_to = std::make_shared<opset10::Transpose>(broadcasted_eltwise2, transpose_to_const);

    auto broadcasted_eltwise_weights_short = create_constant_with_zeros(broadcastedEltwiseShapeShort, {{}, {0, 1}, {}});
    auto broadcasted_eltwise_short = std::make_shared<opset10::Add>(transpose_to, broadcasted_eltwise_weights_short);
    broadcasted_eltwise_short->set_friendly_name("Eltwise broadcasted smaller rank");

    auto broadcasted_eltwise_short2 =
        std::make_shared<opset10::Add>(broadcasted_eltwise_weights_short, broadcasted_eltwise_short);
    broadcasted_eltwise_short2->set_friendly_name("Eltwise broadcasted smaller rank swapped");

    auto transpose_from = std::make_shared<opset10::Transpose>(broadcasted_eltwise_short2, transpose_to_const);

    // Constants should be zero as broadcasted values!
    auto dummy_eltwise_weights = std::make_shared<opset10::Constant>(element::f32, broadcastedEltwiseShape, 0);
    auto dummy_eltwise_inputs = std::make_shared<opset10::Constant>(element::f32, broadcastedEltwiseShape, 0);
    auto dummy_eltwise = std::make_shared<opset10::Add>(dummy_eltwise_inputs, dummy_eltwise_weights);

    auto broadcasted_eltwise_without_weights_mask = std::make_shared<opset10::Add>(transpose_from, dummy_eltwise);
    broadcasted_eltwise_without_weights_mask->set_friendly_name("Eltwise broadcasted same rank without weights mask");

    auto broadcasted_eltwise_without_weights_mask2 =
        std::make_shared<opset10::Add>(dummy_eltwise, broadcasted_eltwise_without_weights_mask);
    broadcasted_eltwise_without_weights_mask2->set_friendly_name(
        "Eltwise broadcasted same rank without weights mask swapped");

    auto left_weights = create_constant_with_zeros(weightsLeftShape, {{}, {1, 2}});

    auto mul_left = std::make_shared<opset10::MatMul>(broadcasted_eltwise_without_weights_mask2, left_weights);

    auto flatten_const = opset10::Constant::create(element::i64, {2}, {2, 36});
    auto flatten = std::make_shared<opset10::Reshape>(mul_left, flatten_const, true);

    auto last_mul_const = create_constant_with_zeros({36, 2}, {{}, {0}});
    auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

    model = std::make_shared<ov::Model>(OutputVector{last_mul}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);

        auto right_weights = create_constant_with_zeros({weightsShape[0], weightsShape[1] - 4}, {{}, {1}});
        auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

        auto eltwise_mul_const =
            create_constant_with_zeros({EltwiseShape[0], EltwiseShape[1], EltwiseShape[2] - 4}, {{}, {1}, {}});
        auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_const);

        auto reshape_const = opset10::Constant::create(element::i64, {4}, {0, 0, 4, 2});
        auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

        auto transpose_const = opset10::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

        auto broadcasted_eltwise_weights = create_constant_with_zeros(broadcastedEltwiseShape, {{0}, {}, {}, {}});
        auto broadcasted_eltwise = std::make_shared<opset10::Add>(transpose, broadcasted_eltwise_weights);

        auto broadcasted_eltwise2 = std::make_shared<opset10::Add>(broadcasted_eltwise_weights, broadcasted_eltwise);
        broadcasted_eltwise2->set_friendly_name("Eltwise broadcasted same rank swapped");

        auto transpose_to_const = opset10::Constant::create(element::i64, {4}, {1, 0, 2, 3});
        auto transpose_to = std::make_shared<opset10::Transpose>(broadcasted_eltwise2, transpose_to_const);

        auto broadcasted_eltwise_weights_short =
            create_constant_with_zeros(broadcastedEltwiseShapeShort, {{}, {0, 1}, {}});
        auto broadcasted_eltwise_short =
            std::make_shared<opset10::Add>(transpose_to, broadcasted_eltwise_weights_short);

        auto broadcasted_eltwise_short2 =
            std::make_shared<opset10::Add>(broadcasted_eltwise_weights_short, broadcasted_eltwise_short);

        auto transpose_from = std::make_shared<opset10::Transpose>(broadcasted_eltwise_short2, transpose_to_const);

        auto dummy_eltwise_weights = std::make_shared<opset10::Constant>(element::f32, broadcastedEltwiseShape, 0);
        auto dummy_eltwise_inputs = std::make_shared<opset10::Constant>(element::f32, broadcastedEltwiseShape, 0);
        auto dummy_eltwise = std::make_shared<opset10::Add>(dummy_eltwise_inputs, dummy_eltwise_weights);

        auto broadcasted_eltwise_without_weights_mask = std::make_shared<opset10::Add>(transpose_from, dummy_eltwise);

        auto broadcasted_eltwise_without_weights_mask2 =
            std::make_shared<opset10::Add>(dummy_eltwise, broadcasted_eltwise_without_weights_mask);

        auto left_weights = create_constant_with_zeros({weightsLeftShape[0], weightsLeftShape[1] - 2}, {{}, {}});

        auto mul_left = std::make_shared<opset10::MatMul>(broadcasted_eltwise_without_weights_mask2, left_weights);

        auto flatten_const = opset10::Constant::create(element::i64, {2}, {2, 8});
        auto flatten = std::make_shared<opset10::Reshape>(mul_left, flatten_const, true);

        auto last_mul_const = create_constant_with_zeros({8, 2}, {{}, {0}});
        auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

        model_ref = std::make_shared<ov::Model>(OutputVector{last_mul}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE) {
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksBroadcastedEltwise.svg")
            .run_on_model(model);
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }
    compare_masks(*getMask(right_weights.get_node_shared_ptr()->output(0)), Mask({{}, {2, 3, 4, 5}}));
    compare_masks(*getMask(mul_right->output(0)), Mask({{}, {}, {2, 3, 4, 5}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask({{}, {}, {1, 2}, {}}));
    compare_masks(*getMask(reshape->output(0)), Mask({{}, {}, {1, 2}, {}}));

    compare_masks(*getMask(transpose->output(0)), Mask({{}, {1, 2}, {}, {}}));
    compare_masks(*getMask(broadcasted_eltwise->output(0)), Mask({{}, {1, 2}, {}, {}}));
    compare_masks(*getMask(broadcasted_eltwise2->output(0)), Mask({{}, {1, 2}, {}, {}}));

    compare_masks(*getMask(transpose_to), Mask{{1, 2}, {}, {}, {}});
    compare_masks(*getMask(broadcasted_eltwise_weights_short), Mask{{}, {}, {}});
    compare_masks(*getMask(broadcasted_eltwise_short->output(0)), Mask{{1, 2}, {}, {}, {}});
    compare_masks(*getMask(broadcasted_eltwise_short2->output(0)), Mask{{1, 2}, {}, {}, {}});
    compare_masks(*getMask(transpose_from), Mask{{}, {1, 2}, {}, {}});

    compare_masks(*getMask(broadcasted_eltwise_without_weights_mask->output(0)), Mask({{}, {1, 2}, {}, {}}));
    compare_masks(*getMask(broadcasted_eltwise_without_weights_mask2->output(0)), Mask({{}, {1, 2}, {}, {}}));

    compare_masks(*getMask(left_weights.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2}}));
    compare_masks(*getMask(mul_left->output(0)), Mask({{}, {1, 2}, {}, {1, 2}}));

    auto ref_flatten_mask = Mask();
    auto ref_dim = std::set<uint64_t>();
    for (uint64_t i = 6; i < 18; ++i)
        ref_dim.insert(i);
    for (uint64_t i = 0; i < 6; ++i)
        for (auto& zero_dim : {1, 2, 4, 5})
            ref_dim.insert(i * 6 + zero_dim);

    ref_flatten_mask.push_back({});
    ref_flatten_mask.push_back(ref_dim);

    compare_masks(*getMask(flatten_const->output(0)), ref_flatten_mask);
    compare_masks(*getMask(flatten->output(0)), ref_flatten_mask);

    auto ref_last_mul_mask = Mask();
    ref_last_mul_mask.push_back(ref_dim);
    ref_last_mul_mask.push_back({});

    compare_masks(*getMask(last_mul_const.get_node_shared_ptr()->output(0)), ref_last_mul_mask);
    compare_masks(*getMask(last_mul->output(0)), Mask({{}, {}}));

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, MaskPropagationComplexReshape) {
    auto inputShapes = PartialShape{1, 6, 3};
    auto weightsShape = Shape{3, 60};
    auto unsquizedShape = Shape{1, 6, 2, 3, 5, 2};  // Restriction: shouldn't be
                                                    // broadcasted dim in column reshaped shape.
                                                    // Leads to pruning of all colums
    auto squizedShape = Shape{1, 6, 60};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    /* 1 -> 0 ch, shoudn't be pruned
       2, 3 -> 1 ch
       4, 5 -> 2 ch */
    auto dims_to_remain = std::vector<uint64_t>{7, 56};
    auto mul_init_mask = Mask(2);
    for (int i = 0; i < 60; ++i)
        mul_init_mask.at(1).insert(i);

    auto mul_weights = create_constant_with_zeros(weightsShape, mul_init_mask);
    auto mul = std::make_shared<opset10::MatMul>(input, mul_weights);

    auto reshape_to_constant = opset10::Constant::create(element::i64, {6}, unsquizedShape);
    auto reshape_to = std::make_shared<opset10::Reshape>(mul, reshape_to_constant, true);

    auto unsquized_eltwise_const_vec = std::vector<float>(360, 0);
    for (size_t i = 0; i < 6; ++i)
        for (auto& elem : dims_to_remain)
            unsquized_eltwise_const_vec[i * 60 + elem] = 1;

    auto unsquized_eltwise_const = opset10::Constant::create(element::f32, unsquizedShape, unsquized_eltwise_const_vec);
    auto unsquized_eltwise = std::make_shared<opset10::Add>(reshape_to, unsquized_eltwise_const);

    auto shape_of = std::make_shared<opset10::ShapeOf>(mul);
    auto reshape_from = std::make_shared<opset10::Reshape>(unsquized_eltwise, shape_of, true);

    for (auto& dim_to_remain : dims_to_remain)
        mul_init_mask.at(1).erase(dim_to_remain);

    auto mul_2_weights = create_constant_with_zeros({weightsShape[1], weightsShape[1]}, mul_init_mask);
    auto mul_2 = std::make_shared<opset10::MatMul>(reshape_from, mul_2_weights);

    auto reshape_to_constant_2 = opset10::Constant::create(element::i64, {6}, unsquizedShape);
    auto reshape_to_2 = std::make_shared<opset10::Reshape>(mul_2, reshape_to_constant_2, true);

    auto shape_of_2 = std::make_shared<opset10::ShapeOf>(mul_2);
    auto reshape_from_2 = std::make_shared<opset10::Reshape>(reshape_to_2, shape_of_2, true);

    // for (auto & elem : dims_to_remain)
    //    mul_init_mask.at(1).erase(elem);

    mul_init_mask.insert(mul_init_mask.begin(), std::set<uint64_t>{});
    auto squized_eltwise_const = create_constant_with_zeros(squizedShape, mul_init_mask);
    auto squized_eltwise = std::make_shared<opset10::Add>(reshape_from_2, squized_eltwise_const);

    auto mul_last_weights = create_constant_with_zeros({weightsShape[1], 1}, {{}, {}});
    auto mul_last = std::make_shared<opset10::MatMul>(squized_eltwise, mul_last_weights);

    model = std::make_shared<ov::Model>(OutputVector{mul_last}, ParameterVector{input});
    const auto remained_dims = std::vector<uint64_t>{6, 7, 26, 27, 36, 37, 56, 57};
    {
        const auto elements_remain = remained_dims.size();
        auto zero_idxs = std::vector<uint64_t>{0, 2, 3, 4, 5, 7};
        auto zero_idxs_set = std::set<uint64_t>(zero_idxs.begin(), zero_idxs.end());
        auto unsquized_const = std::vector<uint64_t>();
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < remained_dims.size(); ++j)
                if (std::find(zero_idxs.begin(), zero_idxs.end(), j) != zero_idxs.end())
                    unsquized_const.push_back(0);
                else
                    unsquized_const.push_back(1);

        auto mul_weights_mask = Mask(2);
        mul_weights_mask.at(1) = zero_idxs_set;
        auto mul_weights = create_constant_with_zeros({weightsShape[0], elements_remain}, {{0, 1, 2}, {}});
        auto mul = std::make_shared<opset10::MatMul>(input, mul_weights);

        auto squized_shape = Shape{1, 6, 2, 2, 1, 2};
        auto reshape_to_constant = opset10::Constant::create(element::i64, {6}, squized_shape);
        auto reshape_to = std::make_shared<opset10::Reshape>(mul, reshape_to_constant, true);

        auto unsquized_eltwise_const = opset10::Constant::create(element::f32, squized_shape, unsquized_const);
        auto unsquized_eltwise = std::make_shared<opset10::Add>(reshape_to, unsquized_eltwise_const);

        auto shape_of = std::make_shared<opset10::ShapeOf>(mul);
        auto reshape_from = std::make_shared<opset10::Reshape>(unsquized_eltwise, shape_of, true);

        auto mul_2_weights = create_constant_with_zeros({elements_remain, elements_remain}, mul_weights_mask);
        auto mul_2 = std::make_shared<opset10::MatMul>(reshape_from, mul_2_weights);

        auto reshape_to_constant_2 = opset10::Constant::create(element::i64, {6}, squized_shape);
        auto reshape_to_2 = std::make_shared<opset10::Reshape>(mul_2, reshape_to_constant_2, true);

        auto shape_of_2 = std::make_shared<opset10::ShapeOf>(mul_2);
        auto reshape_from_2 = std::make_shared<opset10::Reshape>(reshape_to_2, shape_of_2, true);

        auto squized_eltwise_mask = Mask(3);
        squized_eltwise_mask.at(2) = zero_idxs_set;
        auto squized_eltwise_const = create_constant_with_zeros({1, 6, elements_remain}, squized_eltwise_mask);
        auto squized_eltwise = std::make_shared<opset10::Add>(reshape_from_2, squized_eltwise_const);

        auto last_mul_mask = Mask(2);
        last_mul_mask.at(0) = zero_idxs_set;
        auto mul_last_weights = create_constant_with_zeros({elements_remain, 1}, last_mul_mask);
        auto mul_last = std::make_shared<opset10::MatMul>(squized_eltwise, mul_last_weights);

        model_ref = std::make_shared<ov::Model>(OutputVector{mul_last}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE) {
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationComplexReshape.svg").run_on_model(model);
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    auto squized_mask = Mask(2);
    auto squized_dim = std::set<uint64_t>();
    for (size_t i = 0; i < 60; ++i)
        squized_dim.insert(i);

    for (auto& elem : remained_dims)
        squized_dim.erase(elem);

    squized_mask.at(1) = squized_dim;

    compare_masks(*getMask(mul_weights), squized_mask);

    squized_mask.insert(squized_mask.begin(), std::set<uint64_t>());
    compare_masks(*getMask(mul->output(0)), squized_mask);
    compare_masks(*getMask(reshape_to_constant->output(0)), Mask{{}, {}, {}, {1}, {0, 1, 2, 4}, {}});
    compare_masks(*getMask(reshape_to->output(0)), Mask{{}, {}, {}, {1}, {0, 1, 2, 4}, {}});
    compare_masks(*getMask(unsquized_eltwise_const), Mask{{}, {}, {}, {1}, {0, 1, 2, 4}, {}});
    compare_masks(*getMask(unsquized_eltwise->output(0)), Mask{{}, {}, {}, {1}, {0, 1, 2, 4}, {}});
    compare_masks(*getMask(reshape_from->output(0)), squized_mask);

    auto mul_2_mask = Mask(2);
    for (short i = 0; i < 2; ++i)
        mul_2_mask.at(i) = squized_dim;

    compare_masks(*getMask(mul_2_weights), mul_2_mask);
    compare_masks(*getMask(mul_2->output(0)), squized_mask);
    compare_masks(*getMask(reshape_to_constant_2->output(0)), Mask{{}, {}, {}, {1}, {0, 1, 2, 4}, {}});
    compare_masks(*getMask(reshape_to_2->output(0)), Mask{{}, {}, {}, {1}, {0, 1, 2, 4}, {}});
    compare_masks(*getMask(reshape_from_2->output(0)), squized_mask);
    compare_masks(*getMask(squized_eltwise_const), squized_mask);
    compare_masks(*getMask(squized_eltwise->output(0)), squized_mask);
    auto last_mul_mask = Mask(2);
    last_mul_mask.at(0) = squized_dim;
    compare_masks(*getMask(mul_last_weights), last_mul_mask);
    compare_masks(*getMask(mul_last->output(0)), Mask{{}, {}, {}});

    {
        // VisualizeTree modifier helps to print Masks and mark nodes with masks
        auto modifier = [](const Node& node, std::vector<std::string>& attributes) {
            std::stringstream ss;
            size_t index{0};
            for (const auto& output : node.outputs()) {
                if (const auto& mask = getMask(output)) {
                    if (!mask->all_dims_are_empty()) {
                        attributes.emplace_back("color=green");
                        attributes.emplace_back("penwidth=2");
                    }
                    ss << "Mask(" << index << ") : " << *mask << "\\n";
                }
                index++;
            }
            if (!ss.str().empty()) {
                auto label = std::find_if(attributes.begin(), attributes.end(), [](const std::string& value) {
                    return value.find("label=") != std::string::npos;
                });
                if (label != attributes.end()) {
                    label->pop_back();
                    *label += "\n" + ss.str() + "\"";
                } else {
                    attributes.push_back("label=\"" + ss.str() + "\"");
                }
            }
        };

        manager.register_pass<ov::pass::ShrinkWeights>();
        if (VISUALIZE_TESTS_TREE) {
            manager.register_pass<pass::VisualizeTree>(
                std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationComplexReshapeWithMasks.svg",
                modifier);
        }
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_P(TransformationTestsBoolParamF, MaskPropagationReshapedPassThroughP) {
    auto inputShapes = PartialShape{1, 6, 3};
    auto weightsShape = Shape{3, 12};
    auto EltwiseShape = Shape{1, 6, 12};
    auto weightsLeftShape = Shape{6, 3};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    /* 1 -> 0 ch, shoudn't be pruned
       2, 3 -> 1 ch
       4, 5 -> 2 ch */
    auto left_weights = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3, 4, 5}});
    auto mul_left = std::make_shared<opset10::MatMul>(input, left_weights);

    auto eltwise_mul_const = create_constant_with_zeros(EltwiseShape, {{}, {1}, {}});
    auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_left, eltwise_mul_const);

    auto reshape_const = opset10::Constant::create(element::i64, {4}, {0, 0, 6, 2});
    auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

    auto transpose_const = opset10::Constant::create(element::i64, {4}, {0, 2, 3, 1});
    auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

    auto rev_flat_const = opset10::Constant::create(element::i64, {2}, {12, 6});
    auto rev_flat = std::make_shared<opset10::Reshape>(transpose, rev_flat_const, true);

    auto unary_op = std::make_shared<opset10::Softmax>(rev_flat);

    auto add_shape_of = GetParam();
    std::shared_ptr<Node> reshape_recover_shape;
    if (add_shape_of)
        reshape_recover_shape = std::make_shared<opset10::ShapeOf>(transpose);
    else
        reshape_recover_shape = opset10::Constant::create(element::i64, {4}, {1, 6, 2, 6});

    auto reshape_recover = std::make_shared<opset10::Reshape>(unary_op, reshape_recover_shape, true);

    auto right_weights = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3, 4, 5}});
    auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

    auto eltwise_mul_right_const = create_constant_with_zeros(EltwiseShape, {{}, {1}, {}});
    auto eltwise_mul_right = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_right_const);

    auto reshape_const_right = opset10::Constant::create(element::i64, {4}, {0, 0, 6, 2});
    auto reshape_right = std::make_shared<opset10::Reshape>(eltwise_mul_right, reshape_const_right, true);

    auto transpose_const_right = opset10::Constant::create(element::i64, {4}, {0, 2, 1, 3});
    auto transpose_right = std::make_shared<opset10::Transpose>(reshape_right, transpose_const_right);

    auto mul_v = std::make_shared<opset10::MatMul>(reshape_recover, transpose_right);

    auto flatten_const = opset10::Constant::create(element::i64, {2}, {1, 24});
    auto flatten = std::make_shared<opset10::Reshape>(mul_v, flatten_const, true);

    auto last_mul_const = create_constant_with_zeros({24, 2}, {{}, {0}});
    auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

    model = std::make_shared<ov::Model>(OutputVector{last_mul}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto left_weights = create_constant_with_zeros({weightsShape[0], weightsShape[1] - 4}, {{}, {}});
        auto mul_left = std::make_shared<opset10::MatMul>(input, left_weights);

        auto eltwise_mul_const =
            create_constant_with_zeros({EltwiseShape[0], EltwiseShape[1], EltwiseShape[2] - 4}, {{}, {1}, {}});
        auto eltwise_mul = std::make_shared<opset10::Multiply>(mul_left, eltwise_mul_const);

        auto reshape_const = opset10::Constant::create(element::i64, {4}, {0, 0, 4, 2});
        auto reshape = std::make_shared<opset10::Reshape>(eltwise_mul, reshape_const, true);

        auto transpose_const = opset10::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset10::Transpose>(reshape, transpose_const);

        auto rev_flat_const = opset10::Constant::create(element::i64, {2}, {8, 6});
        auto rev_flat = std::make_shared<opset10::Reshape>(transpose, rev_flat_const, true);

        auto unary_op = std::make_shared<opset10::Softmax>(rev_flat);

        std::shared_ptr<Node> reshape_recover_shape;
        if (add_shape_of)
            reshape_recover_shape = std::make_shared<opset10::ShapeOf>(transpose);
        else
            reshape_recover_shape = opset10::Constant::create(element::i64, {4}, {1, 4, 2, 6});

        auto reshape_recover = std::make_shared<opset10::Reshape>(unary_op, reshape_recover_shape, true);

        auto right_weights = create_constant_with_zeros({weightsShape[0], weightsShape[1] - 4}, {{}, {}});
        auto mul_right = std::make_shared<opset10::MatMul>(input, right_weights);

        auto eltwise_mul_right_const =
            create_constant_with_zeros({EltwiseShape[0], EltwiseShape[1], EltwiseShape[2] - 4}, {{}, {1}, {}});
        auto eltwise_mul_right = std::make_shared<opset10::Multiply>(mul_right, eltwise_mul_right_const);

        auto reshape_const_right = opset10::Constant::create(element::i64, {4}, {0, 0, 4, 2});
        auto reshape_right = std::make_shared<opset10::Reshape>(eltwise_mul_right, reshape_const_right, true);

        auto transpose_const_right = opset10::Constant::create(element::i64, {4}, {0, 2, 1, 3});
        auto transpose_right = std::make_shared<opset10::Transpose>(reshape_right, transpose_const_right);

        auto mul_v = std::make_shared<opset10::MatMul>(reshape_recover, transpose_right);

        auto flatten_const = opset10::Constant::create(element::i64, {2}, {1, 16});
        auto flatten = std::make_shared<opset10::Reshape>(mul_v, flatten_const, true);

        auto last_mul_const = create_constant_with_zeros({16, 2}, {{}, {0}});
        auto last_mul = std::make_shared<opset10::MatMul>(flatten, last_mul_const);

        model_ref = std::make_shared<ov::Model>(OutputVector{last_mul}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE) {
        auto postfix = (add_shape_of) ? "True" : "False";
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapedPassThroughP" + postfix + ".svg")
            .run_on_model(model);
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(left_weights.get_node_shared_ptr()->output(0)), Mask{{}, {2, 3, 4, 5}});
    compare_masks(*getMask(mul_left->output(0)), Mask{{}, {}, {2, 3, 4, 5}});
    compare_masks(*getMask(eltwise_mul_const.get_node_shared_ptr()->output(0)), Mask{{}, {}, {2, 3, 4, 5}});
    compare_masks(*getMask(eltwise_mul->output(0)), Mask{{}, {}, {2, 3, 4, 5}});
    compare_masks(*getMask(reshape_const->output(0)), Mask{{{}, {}, {1, 2}, {}}});
    compare_masks(*getMask(reshape->output(0)), Mask{{{}, {}, {1, 2}, {}}});
    compare_masks(*getMask(transpose->output(0)), Mask{{}, {1, 2}, {}, {}});
    compare_masks(*getMask(rev_flat_const->output(0)), Mask{{2, 3, 4, 5}, {}});
    compare_masks(*getMask(rev_flat->output(0)), Mask{{2, 3, 4, 5}, {}});
    compare_masks(*getMask(unary_op->output(0)), Mask{{2, 3, 4, 5}, {}});
    compare_masks(*getMask(reshape_recover->output(0)), Mask{{}, {1, 2}, {}, {}});
    compare_masks(*getMask(right_weights.get_node_shared_ptr()->output(0)), Mask({{}, {2, 3, 4, 5}}));
    compare_masks(*getMask(mul_right->output(0)), Mask{{}, {}, {2, 3, 4, 5}});
    compare_masks(*getMask(eltwise_mul_right_const.get_node_shared_ptr()->output(0)), Mask{{}, {}, {2, 3, 4, 5}});
    compare_masks(*getMask(eltwise_mul_right->output(0)), Mask{{}, {}, {2, 3, 4, 5}});
    compare_masks(*getMask(reshape_const_right->output(0)), Mask{{}, {}, {1, 2}, {}});
    compare_masks(*getMask(reshape_right->output(0)), Mask{{}, {}, {1, 2}, {}});
    compare_masks(*getMask(transpose_right->output(0)), Mask{{}, {1, 2}, {}, {}});
    compare_masks(*getMask(mul_v->output(0)), Mask{{}, {1, 2}, {}, {}});

    auto ref_flatten_mask = Mask();
    auto ref_dim = std::set<uint64_t>();
    for (uint64_t i = 4; i < 12; ++i)
        ref_dim.insert(i);
    ref_flatten_mask.push_back({});
    ref_flatten_mask.push_back(ref_dim);

    compare_masks(*getMask(flatten_const->output(0)), ref_flatten_mask);
    compare_masks(*getMask(flatten->output(0)), ref_flatten_mask);

    ref_flatten_mask[0] = ref_flatten_mask[1];
    ref_flatten_mask[1] = {};

    compare_masks(*getMask(last_mul_const.get_node_shared_ptr()->output(0)), ref_flatten_mask);
    compare_masks(*getMask(last_mul->output(0)), Mask{{}, {}});

    // VisualizeTree modifier helps to print Masks and mark nodes with masks
    auto modifier = [](const Node& node, std::vector<std::string>& attributes) {
        std::stringstream ss;
        size_t index{0};
        for (const auto& output : node.outputs()) {
            if (const auto& mask = getMask(output)) {
                if (!mask->all_dims_are_empty()) {
                    attributes.emplace_back("color=green");
                    attributes.emplace_back("penwidth=2");
                }
                ss << "Mask(" << index << ") : " << *mask << "\\n";
            }
            index++;
        }
        if (!ss.str().empty()) {
            auto label = std::find_if(attributes.begin(), attributes.end(), [](const std::string& value) {
                return value.find("label=") != std::string::npos;
            });
            if (label != attributes.end()) {
                label->pop_back();
                *label += "\n" + ss.str() + "\"";
            } else {
                attributes.push_back("label=\"" + ss.str() + "\"");
            }
        }
    };

    manager.register_pass<ov::pass::ShrinkWeights>();
    if (VISUALIZE_TESTS_TREE) {
        auto postfix = (add_shape_of) ? "True" : "False";
        manager.register_pass<pass::VisualizeTree>(
            std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReverseFlattenWithMasks" + postfix + ".svg",
            modifier);
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_P(TransformationTestsBoolParamF, MaskPropagationBroadcastedSameRankEltwiseSwappedLayoutP) {
    constexpr int64_t a(3), b(4), c(5), d(6);
    auto inputShapes = PartialShape{1, a, b};
    auto weightsShape = Shape{b, c};
    auto weightsShape2 = Shape{c, d};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto mul_const = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3}});
    auto mul = std::make_shared<opset10::MatMul>(input, mul_const);

    auto mul_last_const = create_constant_with_zeros(weightsShape2, {{}, {}});
    auto mult_const = opset10::Constant::create(element::f32, {1, 1}, {5});

    std::shared_ptr<Node> mult;
    const auto reverse_mul = GetParam();
    if (reverse_mul)
        mult = std::make_shared<opset10::Multiply>(mult_const, mul_last_const);
    else
        mult = std::make_shared<opset10::Multiply>(mul_last_const, mult_const);

    auto mul_last = std::make_shared<opset10::MatMul>(mul, mult);

    model = std::make_shared<ov::Model>(OutputVector{mul_last}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
        auto mul_const = create_constant_with_zeros({weightsShape[0], weightsShape[1] - 3}, {{}, {}});
        auto mul = std::make_shared<opset10::MatMul>(input, mul_const);

        auto mul_last_const = create_constant_with_zeros({weightsShape2[0] - 3, weightsShape2[1]}, {{}, {}});
        auto mult_const = opset10::Constant::create(element::f32, {1, 1}, {5});

        std::shared_ptr<Node> mult;
        if (reverse_mul)
            mult = std::make_shared<opset10::Multiply>(mult_const, mul_last_const);
        else
            mult = std::make_shared<opset10::Multiply>(mul_last_const, mult_const);

        auto mul_last = std::make_shared<opset10::MatMul>(mul, mult);

        model_ref = std::make_shared<ov::Model>(OutputVector{mul_last}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE) {
        auto postfix = (reverse_mul) ? "True" : "False";
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) +
                            "MaskPropagationBroadcastedSameRankEltwiseSwappedLayoutP" + postfix + ".svg")
            .run_on_model(model);
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(mul_const), Mask{{}, {1, 2, 3}});
    compare_masks(*getMask(mul->output(0)), Mask{{}, {}, {1, 2, 3}});
    compare_masks(*getMask(mul_last_const), Mask{{1, 2, 3}, {}});
    compare_masks(*getMask(mult_const), Mask{{}, {}});
    compare_masks(*getMask(mult->output(0)), Mask{{1, 2, 3}, {}});
    compare_masks(*getMask(mul_last->output(0)), Mask{{}, {}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, MaskPropagationBroadcastedEltwiseInputAndWeightsBroadcasted) {
    constexpr int64_t a(3), b(4), c(5), d(6);
    auto inputShapes = PartialShape{1, a, b};
    auto weightsShape = Shape{b, c};
    auto weightsShape2 = Shape{c * d, d};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto mul_const = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3}});
    auto mul = std::make_shared<opset10::MatMul>(input, mul_const);

    auto reshape_const = opset10::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, a, c, 1});
    auto reshape = std::make_shared<opset10::Reshape>(mul, reshape_const, true);

    auto mult_const = create_constant_with_zeros({1, a, 1, d}, {{}, {}, {}, {2}});
    auto mult = std::make_shared<opset10::Multiply>(reshape, mult_const);

    auto reshape_from_const = opset10::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, a, c * d});
    auto reshape_from = std::make_shared<opset10::Reshape>(mult, reshape_from_const, true);

    auto mul_last_const = create_constant_with_zeros(weightsShape2, {{}, {}});
    auto mul_last = std::make_shared<opset10::MatMul>(reshape_from, mul_last_const);

    auto model = std::make_shared<ov::Model>(OutputVector{mul_last}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE) {
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) +
                            "MaskPropagationBroadcastedEltwiseInputAndWeightsBroadcasted.svg")
            .run_on_model(model);
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(mul_const), Mask{{}, {}});
    compare_masks(*getMask(mul->output(0)), Mask{{}, {}, {}});
    compare_masks(*getMask(mul_last_const), Mask{{}, {}});
    compare_masks(*getMask(reshape_const), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(reshape->output(0)), Mask{{}, {}, {}, {}});
    check_mask_is_not_exist(getMask(mult_const));
    compare_masks(*getMask(mult->output(0)), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(reshape_from_const), Mask{{}, {}, {}});
    compare_masks(*getMask(reshape_from->output(0)), Mask{{}, {}, {}});
    compare_masks(*getMask(mul_last->output(0)), Mask{{}, {}, {}});

    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);
    }
}

TEST(TransformationTests, MaskPropagationBroadcastedEltwiseWrongBroadcastingMode) {
    constexpr int64_t a(3), b(4), c(5), d(6);
    auto inputShapes = PartialShape{5, a, b};
    auto weightsShape = Shape{b, c};
    auto weightsShape2 = Shape{c, d};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);
    auto mul_const = create_constant_with_zeros(weightsShape, {{}, {1, 2, 3}});
    auto mul = std::make_shared<opset10::MatMul>(input, mul_const);

    auto mult_const = create_constant_with_zeros({c, 1, 1}, {{0, 1, 2, 3, 4}, {}});

    auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0);
    auto mult = std::make_shared<opset10::Multiply>(mul, mult_const, autob);

    auto mul_last_const = create_constant_with_zeros(weightsShape2, {{}, {}});
    auto mul_last = std::make_shared<opset10::MatMul>(mult, mul_last_const);

    auto model = std::make_shared<ov::Model>(OutputVector{mul_last}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE) {
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) +
                            "MaskPropagationBroadcastedEltwiseWrongBroadcastingMode.svg")
            .run_on_model(model);
    }
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(mul_const), Mask{{}, {}});
    compare_masks(*getMask(mul->output(0)), Mask{{}, {}, {}});
    compare_masks(*getMask(mul_last_const), Mask{{}, {}});
    check_mask_is_not_exist(getMask(mult_const));
    compare_masks(*getMask(mult->output(0)), Mask{{}, {}, {}});
    compare_masks(*getMask(mul_last_const), Mask{{}, {}});
    compare_masks(*getMask(mul_last->output(0)), Mask{{}, {}, {}});

    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);
    }
}

TEST_F(TransformationTestsF, MaskPropagationMatMulWithSeveralOutputs) {
    /*Test get_output_shape of a matmul op during masks initialization pass:
     * First matmul has 2 outputs
     * left/right matmuls have 0 outputs
     */
    auto inputShapes = PartialShape{1, 3};
    auto firstMatmulShape = Shape{3, 12};
    auto secondMatmulShape = Shape{12, 2};

    auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);

    auto first_matmul_weights = create_constant_with_zeros(firstMatmulShape, {{}, {1, 2}});
    auto first_matmul = std::make_shared<opset10::MatMul>(input, first_matmul_weights);

    auto left_matmul_weights = create_constant_with_zeros(secondMatmulShape, {{}, {}});
    auto left_matmul = std::make_shared<opset10::MatMul>(first_matmul, left_matmul_weights);

    auto right_matmul_weights = create_constant_with_zeros(secondMatmulShape, {{}, {}});
    auto right_matmul = std::make_shared<opset10::MatMul>(first_matmul, right_matmul_weights);

    model = std::make_shared<ov::Model>(OutputVector{left_matmul, right_matmul}, ParameterVector{input});
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, inputShapes);

        auto first_matmul_weights =
            create_constant_with_zeros({firstMatmulShape[0], firstMatmulShape[1] - 2}, {{}, {}});
        auto first_matmul = std::make_shared<opset10::MatMul>(input, first_matmul_weights);

        auto left_matmul_weights =
            create_constant_with_zeros({secondMatmulShape[0] - 2, secondMatmulShape[1]}, {{}, {}});
        auto left_matmul = std::make_shared<opset10::MatMul>(first_matmul, left_matmul_weights);

        auto right_matmul_weights =
            create_constant_with_zeros({secondMatmulShape[0] - 2, secondMatmulShape[1]}, {{}, {}});
        auto right_matmul = std::make_shared<opset10::MatMul>(first_matmul, right_matmul_weights);

        model_ref = std::make_shared<ov::Model>(OutputVector{left_matmul, right_matmul}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationMatMulWithSeveralOutputs.svg")
            .run_on_model(model);
    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(first_matmul_weights), Mask{{}, {1, 2}});
    compare_masks(*getMask(first_matmul), Mask{{}, {1, 2}});
    compare_masks(*getMask(left_matmul_weights), Mask{{1, 2}, {}});
    compare_masks(*getMask(left_matmul), Mask{{}, {}});
    compare_masks(*getMask(right_matmul_weights), Mask{{1, 2}, {}});
    compare_masks(*getMask(right_matmul), Mask{{}, {}});

    manager.register_pass<ov::pass::ShrinkWeights>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, CheckReshapeWithNoConstInShape) {
    /* Checks condition that `get_constant_from_node` is not nullptr in `is_static_reshape_op`*/
    auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 1});
    auto input_shape = std::make_shared<opset10::Parameter>(element::i64, Shape{3});
    auto reshape = std::make_shared<opset10::Reshape>(input, input_shape, true);
    const auto dummy_mask = std::make_shared<Mask>(Mask{{1}});
    setMask(reshape->output(0), dummy_mask);

    auto model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input, input_shape});

    if (VISUALIZE_TESTS_TREE)
        pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "CheckReshapeWithNoConstInShape.svg")
            .run_on_model(model);

    pass::Manager m;
    m.register_pass<ov::pass::ShrinkWeights>();
    m.run_passes(model);
}

INSTANTIATE_TEST_SUITE_P(TransformationTestsBoolParam, TransformationTestsBoolParamF, ::testing::Values(false, true));

TEST_F(TransformationTestsF, PruningWithVariadicSplitOnSecondAxis) {
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});
        auto weights1 = create_constant_with_zeros({16, 3, 1, 1}, {{1, 2, 4, 8, 10, 11}, {}, {}, {}});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split =
            std::make_shared<opset10::VariadicSplit>(conv1,
                                                     opset10::Constant::create(element::i32, {}, {1}),
                                                     opset10::Constant::create(element::i32, Shape{3}, {2, -1, 8}));
        auto weights2 = create_constant_with_zeros({8, 2, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 6, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 8, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        model = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});
    }

    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});

        auto weights1 = opset10::Constant::create(element::f32, {10, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split_lengths =
            std::make_shared<opset10::Subtract>(opset10::Constant::create(element::i32, Shape{3}, {2, -1, 8}),
                                                opset10::Constant::create(element::i32, Shape{3}, {1, -5, 3}));
        auto split = std::make_shared<opset10::VariadicSplit>(conv1,
                                                              opset10::Constant::create(element::i32, {}, {1}),
                                                              split_lengths);
        auto weights2 = create_constant_with_zeros({8, 1, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 4, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(NodeVector{conv2, conv3, conv4}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::Pruning>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PruningWithVariadicSplitOnFirstAxis) {
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{10, 3, 8, 8});
        auto weights1 = create_constant_with_zeros({16, 3, 1, 1}, {{1, 2, 4, 8, 10, 11}, {}, {}, {}});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split =
            std::make_shared<opset10::VariadicSplit>(conv1,
                                                     opset10::Constant::create(element::i32, {}, {-4}),
                                                     opset10::Constant::create(element::i32, Shape{3}, {-1, 5, 3}));
        auto weights2 = create_constant_with_zeros({8, 16, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 16, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 16, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        model = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});
    }

    {
        // create reference function
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{10, 3, 8, 8});
        auto weights1 = opset10::Constant::create(element::f32, {10, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split =
            std::make_shared<opset10::VariadicSplit>(conv1,
                                                     opset10::Constant::create(element::i32, {}, {-4}),
                                                     opset10::Constant::create(element::i32, Shape{3}, {-1, 5, 3}));
        auto weights2 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::Pruning>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, VariadicSplitMaskPropagationSplitOnSecondAxis) {
    auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});
    auto weights1 = create_constant_with_zeros({16, 3, 1, 1}, {{1, 2, 4, 8, 10, 11}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights1,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto split =
        std::make_shared<opset10::VariadicSplit>(conv1,
                                                 opset10::Constant::create(element::i32, {}, {1}),
                                                 opset10::Constant::create(element::i32, Shape{3}, {-1, 6, 8}));
    auto weights2 = create_constant_with_zeros({8, 2, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                        weights2,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto weights3 = create_constant_with_zeros({8, 6, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                        weights3,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto weights4 = create_constant_with_zeros({8, 8, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                        weights4,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto model = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});

    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights1), Mask{{1, 2, 4, 8, 10, 11}, {}, {}, {}});
    compare_masks(*getMask(conv1), Mask{{}, {1, 2, 4, 8, 10, 11}, {}, {}});
    compare_masks(*getMask(split->output(0)), Mask{{}, {1}, {}, {}});
    compare_masks(*getMask(split->output(1)), Mask{{}, {0, 2}, {}, {}});
    compare_masks(*getMask(split->output(2)), Mask{{}, {0, 2, 3}, {}, {}});
    compare_masks(*getMask(weights2), Mask{{}, {1}, {}, {}});
    compare_masks(*getMask(conv2), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights3), Mask{{}, {0, 2}, {}, {}});
    compare_masks(*getMask(conv3), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights4), Mask{{}, {0, 2, 3}, {}, {}});
    compare_masks(*getMask(conv4), Mask{{}, {}, {}, {}});

    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);

        // create reference function
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});

        auto weights1 = opset10::Constant::create(element::f32, {10, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split_lengths =
            std::make_shared<opset10::Subtract>(opset10::Constant::create(element::i32, Shape{3}, {-1, 6, 8}),
                                                opset10::Constant::create(element::i32, Shape{3}, {-2, 2, 3}));
        auto split = std::make_shared<opset10::VariadicSplit>(conv1,
                                                              opset10::Constant::create(element::i32, {}, {1}),
                                                              split_lengths);
        auto weights2 = create_constant_with_zeros({8, 1, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 4, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto model_ref = std::make_shared<ov::Model>(NodeVector{conv2, conv3, conv4}, ParameterVector{input});

        auto res = compare_functions(model, model_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, VariadicSplitMaskPropagationSplitOnFirstAxis) {
    auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{10, 3, 8, 8});
    auto weights1 = create_constant_with_zeros({16, 3, 1, 1}, {{1, 2, 4, 8, 10, 11}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights1,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto split = std::make_shared<opset10::VariadicSplit>(conv1,
                                                          opset10::Constant::create(element::i32, {}, {0}),
                                                          opset10::Constant::create(element::i32, Shape{3}, {2, 5, 3}));
    auto weights2 = create_constant_with_zeros({8, 16, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                        weights2,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto weights3 = create_constant_with_zeros({8, 16, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                        weights3,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto weights4 = create_constant_with_zeros({8, 16, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                        weights4,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto model = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});

    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights1), Mask{{1, 2, 4, 8, 10, 11}, {}, {}, {}});
    compare_masks(*getMask(conv1), Mask{{}, {1, 2, 4, 8, 10, 11}, {}, {}});
    compare_masks(*getMask(split->output(0)), Mask{{}, {1, 2, 4, 8, 10, 11}, {}, {}});
    compare_masks(*getMask(split->output(1)), Mask{{}, {1, 2, 4, 8, 10, 11}, {}, {}});
    compare_masks(*getMask(split->output(2)), Mask{{}, {1, 2, 4, 8, 10, 11}, {}, {}});
    compare_masks(*getMask(weights2), Mask{{}, {1, 2, 4, 8, 10, 11}, {}, {}});
    compare_masks(*getMask(conv2), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights3), Mask{{}, {1, 2, 4, 8, 10, 11}, {}, {}});
    compare_masks(*getMask(conv3), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights4), Mask{{}, {1, 2, 4, 8, 10, 11}, {}, {}});
    compare_masks(*getMask(conv4), Mask{{}, {}, {}, {}});

    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);

        // create reference function
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{10, 3, 8, 8});
        auto weights1 = opset10::Constant::create(element::f32, {10, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split =
            std::make_shared<opset10::VariadicSplit>(conv1,
                                                     opset10::Constant::create(element::i32, {}, {0}),
                                                     opset10::Constant::create(element::i32, Shape{3}, {2, 5, 3}));
        auto weights2 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto model_ref = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});

        auto res = compare_functions(model, model_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, VariadicSplitMaskPropagationInvalidateMaskOnFirstAndThirdOutput) {
    auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});
    auto weights1 = create_constant_with_zeros({16, 3, 1, 1}, {{1, 2, 4, 8, 10, 11}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights1,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto split = std::make_shared<opset10::VariadicSplit>(conv1,
                                                          opset10::Constant::create(element::i32, {}, {1}),
                                                          opset10::Constant::create(element::i32, Shape{3}, {2, 6, 8}));
    auto weights2 = create_constant_with_zeros({8, 6, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(split->output(1),
                                                        weights2,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto model =
        std::make_shared<ov::Model>(OutputVector{split->output(0), conv2, split->output(2)}, ParameterVector{input});

    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights1), Mask{{2, 4}, {}, {}, {}});
    compare_masks(*getMask(conv1), Mask{{}, {2, 4}, {}, {}});
    compare_masks(*getMask(split->output(0)), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(split->output(1)), Mask{{}, {0, 2}, {}, {}});
    compare_masks(*getMask(split->output(2)), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights2), Mask{{}, {0, 2}, {}, {}});
    compare_masks(*getMask(conv2), Mask{{}, {}, {}, {}});

    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);

        // create reference function
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});

        auto weights1 = opset10::Constant::create(element::f32, {14, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split_lengths =
            std::make_shared<opset10::Subtract>(opset10::Constant::create(element::i32, Shape{3}, {2, 6, 8}),
                                                opset10::Constant::create(element::i32, Shape{3}, {0, 2, 0}));
        auto split = std::make_shared<opset10::VariadicSplit>(conv1,
                                                              opset10::Constant::create(element::i32, {}, {1}),
                                                              split_lengths);
        auto weights2 = create_constant_with_zeros({8, 4, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto model_ref = std::make_shared<ov::Model>(OutputVector{split->output(0), conv2, split->output(2)},
                                                     ParameterVector{input});

        auto res = compare_functions(model, model_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST_F(TransformationTestsF, PruningWithSplitOnSecondAxis) {
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});
        auto weights1 = create_constant_with_zeros({15, 3, 1, 1}, {{1, 2, 4, 8, 10}, {}, {}, {}});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split = std::make_shared<opset10::Split>(conv1, opset10::Constant::create(element::i32, {}, {1}), 3);
        auto weights2 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        model = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});
    }

    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});

        auto weights1 = opset10::Constant::create(element::f32, {10, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split_lengths = opset10::Constant::create(element::i64, Shape{3}, {2, 4, 4});
        auto split = std::make_shared<opset10::VariadicSplit>(conv1,
                                                              opset10::Constant::create(element::i32, {}, {1}),
                                                              split_lengths);
        auto weights2 = create_constant_with_zeros({8, 2, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 4, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 4, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(NodeVector{conv2, conv3, conv4}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::Pruning>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PruningWithSplitOnFirstAxis) {
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{6, 3, 8, 8});
        auto weights1 = create_constant_with_zeros({15, 3, 1, 1}, {{1, 2, 4, 8, 10}, {}, {}, {}});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split = std::make_shared<opset10::Split>(conv1, opset10::Constant::create(element::i32, {}, {0}), 3);
        auto weights2 = create_constant_with_zeros({8, 15, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 15, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 15, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        model = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});
    }

    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{6, 3, 8, 8});

        auto weights1 = opset10::Constant::create(element::f32, {10, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split = std::make_shared<opset10::Split>(conv1, opset10::Constant::create(element::i32, {}, {0}), 3);
        auto weights2 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(NodeVector{conv2, conv3, conv4}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::Pruning>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST(TransformationTests, SplitMaskPropagationSplitOnSecondAxis) {
    auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});
    auto weights1 = create_constant_with_zeros({15, 3, 1, 1}, {{1, 2, 4, 8, 10}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights1,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto split = std::make_shared<opset10::Split>(conv1, opset10::Constant::create(element::i32, {}, {1}), 3);
    auto weights2 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                        weights2,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto weights3 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                        weights3,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto weights4 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                        weights4,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto model = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});

    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights1), Mask{{1, 2, 4, 8, 10}, {}, {}, {}});
    compare_masks(*getMask(conv1), Mask{{}, {1, 2, 4, 8, 10}, {}, {}});
    compare_masks(*getMask(split->output(0)), Mask{{}, {1, 2, 4}, {}, {}});
    compare_masks(*getMask(split->output(1)), Mask{{}, {3}, {}, {}});
    compare_masks(*getMask(split->output(2)), Mask{{}, {0}, {}, {}});
    compare_masks(*getMask(weights2), Mask{{}, {1, 2, 4}, {}, {}});
    compare_masks(*getMask(conv2), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights3), Mask{{}, {3}, {}, {}});
    compare_masks(*getMask(conv3), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights4), Mask{{}, {0}, {}, {}});
    compare_masks(*getMask(conv4), Mask{{}, {}, {}, {}});

    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);

        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});

        auto weights1 = opset10::Constant::create(element::f32, {10, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split_lengths = opset10::Constant::create(element::i64, Shape{3}, {2, 4, 4});
        auto split = std::make_shared<opset10::VariadicSplit>(conv1,
                                                              opset10::Constant::create(element::i32, {}, {1}),
                                                              split_lengths);
        auto weights2 = create_constant_with_zeros({8, 2, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 4, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 4, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto model_ref = std::make_shared<ov::Model>(NodeVector{conv2, conv3, conv4}, ParameterVector{input});

        auto res = compare_functions(model, model_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, SplitMaskPropagationSplitOnFirstAxis) {
    auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{6, 3, 8, 8});
    auto weights1 = create_constant_with_zeros({15, 3, 1, 1}, {{1, 2, 4, 8, 10}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights1,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto split = std::make_shared<opset10::Split>(conv1, opset10::Constant::create(element::i32, {}, {0}), 3);
    auto weights2 = create_constant_with_zeros({8, 15, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                        weights2,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto weights3 = create_constant_with_zeros({8, 15, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                        weights3,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto weights4 = create_constant_with_zeros({8, 15, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                        weights4,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto model = std::make_shared<ov::Model>(OutputVector{conv2, conv3, conv4}, ParameterVector{input});

    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights1), Mask{{1, 2, 4, 8, 10}, {}, {}, {}});
    compare_masks(*getMask(conv1), Mask{{}, {1, 2, 4, 8, 10}, {}, {}});
    compare_masks(*getMask(split->output(0)), Mask{{}, {1, 2, 4, 8, 10}, {}, {}});
    compare_masks(*getMask(split->output(1)), Mask{{}, {1, 2, 4, 8, 10}, {}, {}});
    compare_masks(*getMask(split->output(2)), Mask{{}, {1, 2, 4, 8, 10}, {}, {}});
    compare_masks(*getMask(weights2), Mask{{}, {1, 2, 4, 8, 10}, {}, {}});
    compare_masks(*getMask(conv2), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights3), Mask{{}, {1, 2, 4, 8, 10}, {}, {}});
    compare_masks(*getMask(conv3), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights4), Mask{{}, {1, 2, 4, 8, 10}, {}, {}});
    compare_masks(*getMask(conv4), Mask{{}, {}, {}, {}});

    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);

        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{6, 3, 8, 8});

        auto weights1 = opset10::Constant::create(element::f32, {10, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split = std::make_shared<opset10::Split>(conv1, opset10::Constant::create(element::i32, {}, {0}), 3);
        auto weights2 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(0),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights3 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv3 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights3,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto weights4 = create_constant_with_zeros({8, 10, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv4 = std::make_shared<opset10::Convolution>(split->output(2),
                                                            weights4,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto model_ref = std::make_shared<ov::Model>(NodeVector{conv2, conv3, conv4}, ParameterVector{input});

        auto res = compare_functions(model, model_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, SplitMaskPropagationInvalidateMaskOnFirstAndThirdOutput) {
    auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});
    auto weights1 = create_constant_with_zeros({15, 3, 1, 1}, {{1, 2, 4, 6, 8, 10, 11}, {}, {}, {}});
    auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                        weights1,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto split = std::make_shared<opset10::Split>(conv1, opset10::Constant::create(element::i32, {}, {1}), 3);
    auto weights2 = create_constant_with_zeros({8, 5, 1, 1}, {{1, 2}, {}, {}, {}});
    auto conv2 = std::make_shared<opset10::Convolution>(split->output(1),
                                                        weights2,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto model =
        std::make_shared<ov::Model>(OutputVector{split->output(0), conv2, split->output(2)}, ParameterVector{input});

    {
        pass::Manager m;
        m.register_pass<ov::pass::InitMasks>();
        m.register_pass<ov::pass::PropagateMasks>();
        m.run_passes(model);
    }

    compare_masks(*getMask(weights1), Mask{{6, 8}, {}, {}, {}});
    compare_masks(*getMask(conv1), Mask{{}, {6, 8}, {}, {}});
    compare_masks(*getMask(split->output(0)), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(split->output(1)), Mask{{}, {1, 3}, {}, {}});
    compare_masks(*getMask(split->output(2)), Mask{{}, {}, {}, {}});
    compare_masks(*getMask(weights2), Mask{{}, {1, 3}, {}, {}});
    compare_masks(*getMask(conv2), Mask{{}, {}, {}, {}});

    {
        pass::Manager m;
        m.register_pass<ov::pass::ShrinkWeights>();
        m.run_passes(model);

        // create reference function
        auto input = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 8, 8});

        auto weights1 = opset10::Constant::create(element::f32, {13, 3, 1, 1}, {1});
        auto conv1 = std::make_shared<opset10::Convolution>(input,
                                                            weights1,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto split_lengths = opset10::Constant::create(element::i64, Shape{3}, {5, 3, 5});
        auto split = std::make_shared<opset10::VariadicSplit>(conv1,
                                                              opset10::Constant::create(element::i32, {}, {1}),
                                                              split_lengths);
        auto weights2 = create_constant_with_zeros({8, 3, 1, 1}, {{1, 2}, {}, {}, {}});
        auto conv2 = std::make_shared<opset10::Convolution>(split->output(1),
                                                            weights2,
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
        auto model_ref = std::make_shared<ov::Model>(OutputVector{split->output(0), conv2, split->output(2)},
                                                     ParameterVector{input});

        auto res = compare_functions(model, model_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST_F(TransformationTestsF, PruningReshapeNegativeOne) {
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2, 3});
        auto weights1 = create_constant_with_zeros({1, 3, 12}, {{}, {}, {}});
        auto matmul1 = std::make_shared<opset10::MatMul>(input, weights1);
        auto reshape =
            std::make_shared<opset10::Reshape>(matmul1,
                                               opset10::Constant::create(element::i32, Shape{4}, {0, 2, 6, -1}),
                                               true);

        auto weights2 = create_constant_with_zeros({1, 3, 6}, {{}, {}, {1, 2}});
        auto matmul2 = std::make_shared<opset10::MatMul>(input, weights2);

        auto matmul3 = std::make_shared<opset10::MatMul>(matmul2, reshape);

        model = std::make_shared<ov::Model>(OutputVector{matmul3}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2, 3});
        auto weights1 = create_constant_with_zeros({1, 3, 8}, {{}, {}, {}});
        auto matmul1 = std::make_shared<opset10::MatMul>(input, weights1);
        auto reshape =
            std::make_shared<opset10::Reshape>(matmul1,
                                               opset10::Constant::create(element::i32, Shape{4}, {0, 2, 4, -1}),
                                               true);

        auto weights2 = create_constant_with_zeros({1, 3, 4}, {{}, {}, {}});
        auto matmul2 = std::make_shared<opset10::MatMul>(input, weights2);

        auto matmul3 = std::make_shared<opset10::MatMul>(matmul2, reshape);

        model_ref = std::make_shared<ov::Model>(OutputVector{matmul3}, ParameterVector{input});
    }
    manager.register_pass<ov::pass::Pruning>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PruningReshapeNegativeOneNonConstantShape) {
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2, 3});
        auto weights1 = create_constant_with_zeros({1, 3, 12}, {{}, {}, {}});
        auto matmul1 = std::make_shared<opset10::MatMul>(input, weights1);

        auto shape = std::make_shared<opset10::ShapeOf>(matmul1);
        auto second_dim = std::make_shared<opset10::Gather>(shape,
                                                            opset10::Constant::create(element::i32, Shape{1}, {1}),
                                                            opset10::Constant::create(element::i32, Shape{1}, {0}));
        auto concat =
            std::make_shared<opset10::Concat>(OutputVector{opset10::Constant::create(element::i64, Shape{1}, {0}),
                                                           second_dim,
                                                           opset10::Constant::create(element::i64, Shape{1}, {6}),
                                                           opset10::Constant::create(element::i64, Shape{1}, {-1})},
                                              0);
        auto reshape = std::make_shared<opset10::Reshape>(matmul1, concat, true);

        auto weights2 = create_constant_with_zeros({1, 3, 6}, {{}, {}, {1, 2}});
        auto matmul2 = std::make_shared<opset10::MatMul>(input, weights2);

        auto matmul3 = std::make_shared<opset10::MatMul>(matmul2, reshape);

        model = std::make_shared<ov::Model>(OutputVector{matmul3}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2, 3});
        auto weights1 = create_constant_with_zeros({1, 3, 8}, {{}, {}, {}});
        auto matmul1 = std::make_shared<opset10::MatMul>(input, weights1);

        auto shape = std::make_shared<opset10::ShapeOf>(matmul1);
        auto second_dim = std::make_shared<opset10::Gather>(shape,
                                                            opset10::Constant::create(element::i32, Shape{1}, {1}),
                                                            opset10::Constant::create(element::i32, Shape{1}, {0}));
        auto concat =
            std::make_shared<opset10::Concat>(OutputVector{opset10::Constant::create(element::i64, Shape{1}, {0}),
                                                           second_dim,
                                                           opset10::Constant::create(element::i64, Shape{1}, {6}),
                                                           opset10::Constant::create(element::i64, Shape{1}, {-1})},
                                              0);
        auto sub = std::make_shared<opset10::Subtract>(concat,
                                                       opset10::Constant::create(element::i64, Shape{4}, {0, 0, 2, 0}));
        auto reshape = std::make_shared<opset10::Reshape>(matmul1, sub, true);

        auto weights2 = create_constant_with_zeros({1, 3, 4}, {{}, {}, {}});
        auto matmul2 = std::make_shared<opset10::MatMul>(input, weights2);

        auto matmul3 = std::make_shared<opset10::MatMul>(matmul2, reshape);

        model_ref = std::make_shared<ov::Model>(OutputVector{matmul3}, ParameterVector{input});
    }
    manager.register_pass<ov::pass::Pruning>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
