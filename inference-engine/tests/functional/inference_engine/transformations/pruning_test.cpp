// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <pruning.hpp>
#include <mask_attribute.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/coordinate_transform.hpp>
#include <ngraph/pass/manager.hpp>
#include <inference_engine.hpp>

using namespace testing;
using namespace ngraph;

void compare_masks(const Mask & mask, const Mask & ref_mask) {
    ASSERT_EQ(mask.size(), ref_mask.size());
    ASSERT_EQ(mask, ref_mask);
}

Output<Node> create_constant_with_zeros(const Shape & shape, const Mask & mask) {
    std::vector<double> values(shape_size(shape), 1);
    for (size_t dim = 0; dim < mask.size(); ++dim) {
        for (const auto & dim_value : mask.at(dim)) {
            Coordinate coord_begin(shape.size(), 0);
            coord_begin[dim] = dim_value;

            Coordinate coord_end(shape);
            coord_end[dim] = dim_value + 1;

            CoordinateTransform iter(shape, coord_begin, coord_end);
            for (const Coordinate & coord : iter) {
                values[iter.index(coord)] = 0;
            }
        }
    }
    return std::make_shared<opset5::Constant>(element::f32, shape, values);
}

TEST(TransformationTests, InitMasksOI) {
    Shape weights_shape{6, 3, 3, 3};
    auto weights = opset5::Constant::create(element::f32, weights_shape, {0});
    pass::InitConstMask({0, 1}).apply(weights);

    compare_masks(*getMask(weights->output(0)), {{0, 1, 2, 3, 4, 5}, {0, 1, 2}, {}, {}});
}

TEST(TransformationTests, InitMasksOutputChannel) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};

    std::vector<double> values(shape_size(weights_shape), 1);
    CoordinateTransform iter(weights_shape, {0, 1, 0, 0}, {6, 2, 3, 3});
    for (const Coordinate & coord : iter) {
        values[iter.index(coord)] = 0;
    }

    auto weights = std::make_shared<opset5::Constant>(element::f32, weights_shape, values);
    pass::InitConstMask({1}).apply(weights);

    compare_masks(*getMask(weights->output(0)), {{}, {1}, {}, {}});
}

// TODO: add test init masks with subgraph
TEST(TransformationTests, TestInitMasks) {
    Shape weights_shape{6, 3, 3, 3};
    Shape input_shape{1, 3, 64, 64};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{input});
    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)), {{1, 2, 3}, {}, {}, {}});
}

TEST(TransformationTests, InitMasksNegative) {
    Shape weights_shape{6, 3, 3, 3};
    auto weights = opset5::Constant::create(element::f32, weights_shape, {0.5});
    pass::InitConstMask({0, 1, 2, 3}).apply(weights);

    compare_masks(*getMask(weights->output(0)), {{}, {}, {}, {}});
}

TEST(TransformationTests, PropagateMasksNegative) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = opset5::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights->output(0)), {{}, {}, {}, {}});
    compare_masks(*getMask(conv->output(0)), {{}, {}, {}, {}});
}

TEST(TransformationTests, PropagateMasksBasic) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_shape2{6, 6, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = opset5::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto add_const = create_constant_with_zeros(Shape{1, 6, 1, 1}, {{}, {1, 2, 3, 4, 5}, {}, {}});
    auto add = std::make_shared<opset5::Add>(relu, add_const);

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2, 3}, {}, {}});
    auto sub = std::make_shared<opset5::Subtract>(add, sub_const);

    auto mul_const = create_constant_with_zeros(Shape{1, 6, 1, 1}, {{}, {4}, {}, {}});
    auto mul = std::make_shared<opset5::Multiply>(sub, mul_const);

    auto weights2 = create_constant_with_zeros(weights_shape2, {{1, 2}, {1, 2, 3}, {}, {}});
    auto conv2 = std::make_shared<opset5::Convolution>(mul, weights2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights->output(0)),  Mask({{1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),     Mask({{}, {1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(add_const), Mask({{}, {1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(mul_const), Mask({{}, {1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{}, {1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{}, {1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(weights2.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),    Mask({{}, {}, {}, {}}));
}

TEST(TransformationTests, PropagateMasksDynamicConvolution) {
    PartialShape input_shape{Dimension::dynamic(), 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_shape2{6, 6, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = opset5::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2, 3}, {}, {}});
    auto sub = std::make_shared<opset5::Subtract>(relu, sub_const);

    auto mul_const = create_constant_with_zeros(Shape{6, 1, 1}, {{2}, {}, {}});
    auto mul = std::make_shared<opset5::Subtract>(sub, mul_const);

    auto weights2 = opset5::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(mul, weights2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights->output(0)),  Mask({{2}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),     Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{2}, {}, {}}));
    compare_masks(*getMask(mul_const), Mask({{2}, {}, {}}));
    compare_masks(*getMask(weights2->output(0)), Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),    Mask({{}, {}, {}, {}}));
}

TEST(TransformationTests, PropagateMasksDynamicGroupConvolution) {
    PartialShape input_shape{Dimension::dynamic(), 3, 64, 64};
    Shape weights_shape{3, 2, 1, 3, 3};
    Shape weights_shape2{6, 1, 1, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = opset5::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset5::GroupConvolution>(input, weights, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2, 3}, {}, {}});
    auto sub = std::make_shared<opset5::Subtract>(relu, sub_const);

    auto mul_const = create_constant_with_zeros(Shape{6, 1, 1}, {{2}, {}, {}});
    auto mul = std::make_shared<opset5::Subtract>(sub, mul_const);

    auto weights2 = opset5::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset5::GroupConvolution>(mul, weights2, Strides(2, 1),
                                                            CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);
}

TEST(TransformationTests, PropagateMasksEmpty) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_shape2{6, 6, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = opset5::Constant::create(element::f32, weights_shape, {1.});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2, 3}, {}, {}});
    auto sub = std::make_shared<opset5::Subtract>(relu, sub_const);

    auto add_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2}, {}, {}});
    auto add = std::make_shared<opset5::Subtract>(sub, add_const);

    auto weights2 = opset5::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(add, weights2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),     Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{}, {}, {}}));
    compare_masks(*getMask(add_const), Mask({{}, {}, {}}));
    compare_masks(*getMask(weights2->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),    Mask({{}, {}, {}, {}}));
}

TEST(TransformationTests, PropagateMaskPassThrough) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_const_1 = create_constant_with_zeros(weights_shape, {{1, 2, 3}, {}, {}, {}});
    weights_const_1.get_node_shared_ptr()->set_friendly_name("weights_1");

    auto conv_1 = std::make_shared<opset5::Convolution>(input, weights_const_1, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    conv_1->set_friendly_name("conv_1");

    // Adding a couple of PassThrough operations
    auto relu = std::make_shared<opset5::Relu>(conv_1);
    relu->set_friendly_name("relu");

    auto clamp = std::make_shared<opset5::Clamp>(relu, 0, 6);
    clamp->set_friendly_name("clamp");

    auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
    auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
    auto pad = std::make_shared<opset5::Pad>(clamp, pads_begin, pads_end, op::PadMode::CONSTANT);
    auto max_pool = std::make_shared<opset5::MaxPool>(pad, Strides{1, 1},
                                                      Shape{0, 0}, Shape{1, 1}, Shape{4, 4});
    max_pool->set_friendly_name("max_pool");

    auto weights2 = opset5::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(max_pool, weights2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights_const_1.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(clamp->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(max_pool->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
}

TEST(TransformationTests, PropagateMasksHardDependencies) {
    Shape input_shape{1, 3, 3, 3};

    auto input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input1->set_friendly_name("input1");

    Shape weights1_shape{6, 3, 3, 3};
    auto weights1 = create_constant_with_zeros(weights1_shape, {{1, 2, 3}, {}, {}, {}});
    weights1.get_node_shared_ptr()->set_friendly_name("weights1");

    auto conv1 = std::make_shared<opset5::Convolution>(input1, weights1, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    conv1->set_friendly_name("conv1");

    auto relu = std::make_shared<opset5::Relu>(conv1);
    relu->set_friendly_name("relu");

    auto input2 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input2->set_friendly_name("input2");

    Shape weights2_shape{6, 3, 3, 3};
    auto weights2 = create_constant_with_zeros(weights2_shape, {{2, 3}, {}, {}, {}});
    weights2.get_node_shared_ptr()->set_friendly_name("weights2");

    auto conv2 = std::make_shared<opset5::Convolution>(input2, weights2, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    conv2->set_friendly_name("conv2");

    auto add1 = std::make_shared<opset5::Add>(conv2, conv1);
    add1->set_friendly_name("add1");

    auto reshape = std::make_shared<opset5::Reshape>(add1, opset5::Constant::create(element::i64, Shape{2}, {1, 6}), true);
    reshape->set_friendly_name("reshape");

    auto matmul = std::make_shared<opset5::MatMul>(reshape, opset5::Constant::create(element::f32, Shape{6, 100}, {1.}));
    matmul->set_friendly_name("matmul");

    auto add2 = std::make_shared<opset5::Add>(conv2, create_constant_with_zeros({6, 1, 1}, {{2}, {}, {}}));
    add2->set_friendly_name("add2");

    Shape weights_shape3{6, 6, 1, 1};
    auto weights3 = opset5::Constant::create(element::f32, weights_shape3, {0});
    weights3->set_friendly_name("weights3");

    auto conv3 = std::make_shared<opset5::Convolution>(add2, weights3, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    conv3->set_friendly_name("conv3");

    auto f = std::make_shared<Function>(NodeVector{matmul, conv3}, ParameterVector{input1, input2});

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(f);

    // TODO: add checks after MatMul/Reshape/Pooling mask propagation is ready
//    compare_masks(*getMask(weights),  Mask({{0, 1, 2, 3, 4, 5}, {}, {}, {}}));
//    compare_masks(*getMask(conv),     Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
//    compare_masks(*getMask(relu),     Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
//    compare_masks(*getMask(weights2), Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
//    compare_masks(*getMask(conv2),    Mask({{}, {}, {}, {}}));
}

TEST(TransformationTests, PropagateMasksQuantizedGroupConvolution) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weights_group_shape{8, 1, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");

    auto weights1 = create_constant_with_zeros(weights_shape, {{0, 1, 2, 3}, {}, {}, {}});
    auto conv1 = std::make_shared<opset5::Convolution>(input, weights1, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto weights_group = opset5::Constant::create(element::i8, weights_group_shape, {0});
    weights_group->set_friendly_name("weights_group");

    auto convert = std::make_shared<opset5::Convert>(weights_group, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3}, {}, {}, {}});

    auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});
    auto mul = std::make_shared<opset5::Multiply>(sub, mul_const);
    mul->set_friendly_name("mul");

    auto reshape = std::make_shared<opset5::Reshape>(mul, opset5::Constant::create(element::i64, Shape{5}, {8, 1, 1, 3, 3}), false);

    auto conv_group = std::make_shared<opset5::GroupConvolution>(conv1, reshape, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});;
    auto add = std::make_shared<opset5::Add>(conv_group, add_const);
    add->set_friendly_name("add");

    auto weights_2 = opset5::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(add, weights_2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(f);

    compare_masks(*getMask(weights1.get_node_shared_ptr()->output(0)), Mask({{0 , 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0 , 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_group->output(0)), Mask({{0 , 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{0 , 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)), Mask({{0 , 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{0 , 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)), Mask({{0 , 1, 2, 3}, {}, {}, {}}));

    compare_masks(*getMask(reshape->output(0)), Mask({{0 , 1, 2, 3}, {}, {}, {}, {}}));

    compare_masks(*getMask(conv_group->output(0)),  Mask({{}, {0 , 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(weights_2->output(0)),  Mask({{}, {0, 1, 2, 3}, {}, {}}));
}

TEST(TransformationTests, PropagateMasksFakeQuantizePerTensor) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_1 = opset5::Constant::create(element::i8, weights_shape, {0});
    weights_1->set_friendly_name("weights_int8_const");

    auto convert = std::make_shared<opset5::Convert>(weights_1, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3}, {}, {}, {}});

    auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});
    auto mul = std::make_shared<opset5::Multiply>(sub, mul_const);
    mul->set_friendly_name("mul");

    auto conv1 = std::make_shared<opset5::Convolution>(input, mul, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    conv1->set_friendly_name("conv1");

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});;
    auto add = std::make_shared<opset5::Add>(conv1, add_const);
    add->set_friendly_name("add");

    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset5::Constant::create(element::f32, Shape{1, 1, 1, 1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{}, {1});
    auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

    auto weights_2 = opset5::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(fq, weights_2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(f);

    compare_masks(*getMask(weights_1->output(0)), Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(add->output(0)),  Mask({{}, {0 , 1, 2, 3, 4},  {}, {}}));

    compare_masks(*getMask(fq->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {}, {}, {}}));
}

TEST(TransformationTests, PropagateMasksFakeQuantizePerChannel) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_1 = opset5::Constant::create(element::i8, weights_shape, {0});
    weights_1->set_friendly_name("weights_int8_const");

    auto convert = std::make_shared<opset5::Convert>(weights_1, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3}, {}, {}, {}});

    auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});
    auto mul = std::make_shared<opset5::Multiply>(sub, mul_const);
    mul->set_friendly_name("mul");

    auto conv1 = std::make_shared<opset5::Convolution>(input, mul, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    conv1->set_friendly_name("conv1");

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});;
    auto add = std::make_shared<opset5::Add>(conv1, add_const);
    add->set_friendly_name("add");

    auto input_low = opset5::Constant::create(element::f32, Shape{1, 8, 1, 1}, {0});
    auto input_high = opset5::Constant::create(element::f32, Shape{1, 8, 1, 1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{8, 1, 1}, {1});
    auto output_high = opset5::Constant::create(element::f32, Shape{8, 1, 1}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

    auto weights_2 = opset5::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(fq, weights_2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights_1->output(0)), Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(add->output(0)),  Mask({{}, {0 , 1, 2, 3, 4},  {}, {}}));

    compare_masks(*getMask(fq->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(fq->input(1).get_source_output()),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(2).get_source_output()),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(3).get_source_output()),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(4).get_source_output()),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
}

TEST(TransformationTests, TestConcatMaskPropagation) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape1{8, 3, 3, 3};
    Shape weights_shape2{16, 3, 3, 3};
    Shape weights_shape3{8, 3, 3, 3};

    Shape weight_shape_out_conv{3, 32, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights_1 = create_constant_with_zeros(weights_shape1, {{0, 1, 2, 3}, {}, {}, {}});
    auto conv1 = std::make_shared<opset5::Convolution>(input, weights_1, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto weights_2 = create_constant_with_zeros(weights_shape2, {{7, 8, 9, 10}, {}, {}, {}});
    auto conv2 = std::make_shared<opset5::Convolution>(input, weights_2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto weights_3 = create_constant_with_zeros(weights_shape3, {{4, 5, 6, 7}, {}, {}, {}});
    auto conv3 = std::make_shared<opset5::Convolution>(input, weights_3, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto concat = std::make_shared<opset5::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

    auto weights_out_conv = create_constant_with_zeros(weight_shape_out_conv, {{}, {}, {}, {}});
    auto conv_out = std::make_shared<opset5::Convolution>(concat, weights_out_conv, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto f = std::make_shared<Function>(NodeVector{conv_out}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights_1.get_node_shared_ptr()->output(0)),  Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0, 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_2.get_node_shared_ptr()->output(0)),  Mask({{7, 8, 9, 10}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {7, 8, 9, 10}, {}, {}}));

    compare_masks(*getMask(weights_3.get_node_shared_ptr()->output(0)),  Mask({{4, 5, 6, 7}, {}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)),  Mask({{}, {4, 5, 6, 7}, {}, {}}));

    compare_masks(*getMask(concat->output(0)),  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
    compare_masks(*getMask(weights_out_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
}


TEST(TransformationTests, TestConcatMaskPropagationUp) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape1{8, 3, 3, 3};
    Shape weights_shape2{16, 3, 3, 3};
    Shape weights_shape3{8, 3, 3, 3};

    Shape weight_shape_out_conv{3, 32, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights_1 = create_constant_with_zeros(weights_shape1, {{0, 1, 2, 3, 4, 5}, {}, {}, {}});
    auto conv1 = std::make_shared<opset5::Convolution>(input, weights_1, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto weights_2 = create_constant_with_zeros(weights_shape2, {{7, 8, 9, 10}, {}, {}, {}});
    auto conv2 = std::make_shared<opset5::Convolution>(input, weights_2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto weights_3 = create_constant_with_zeros(weights_shape3, {{2, 3, 4, 5, 6, 7}, {}, {}, {}});
    auto conv3 = std::make_shared<opset5::Convolution>(input, weights_3, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto concat = std::make_shared<opset5::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

    auto add_const = create_constant_with_zeros(Shape{1, 32, 1, 1}, {{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}});
    auto add = std::make_shared<opset5::Add>(concat, add_const);

    auto weights_out_conv = create_constant_with_zeros(weight_shape_out_conv, {{}, {}, {}, {}});
    auto conv_out = std::make_shared<opset5::Convolution>(add, weights_out_conv, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto f = std::make_shared<Function>(NodeVector{conv_out}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights_1.get_node_shared_ptr()->output(0)),  Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0, 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_2.get_node_shared_ptr()->output(0)),  Mask({{7, 8, 9, 10}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {7, 8, 9, 10}, {}, {}}));

    compare_masks(*getMask(weights_3.get_node_shared_ptr()->output(0)),  Mask({{4, 5, 6, 7}, {}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)),  Mask({{}, {4, 5, 6, 7}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)),  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
    compare_masks(*getMask(add->output(0)),  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));


    compare_masks(*getMask(concat->output(0)),  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
    compare_masks(*getMask(weights_out_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
}


TEST(TransformationTests, TestConcatMaskPropagationUpEmpty) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape1{8, 3, 3, 3};
    Shape weights_shape2{16, 3, 3, 3};
    Shape weights_shape3{8, 3, 3, 3};

    Shape weight_shape_out_conv{3, 32, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights_1 = create_constant_with_zeros(weights_shape1, {{0, 1, 2, 3, 4, 5}, {}, {}, {}});
    auto conv1 = std::make_shared<opset5::Convolution>(input, weights_1, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto weights_2 = create_constant_with_zeros(weights_shape2, {{7, 8, 9, 10}, {}, {}, {}});
    auto conv2 = std::make_shared<opset5::Convolution>(input, weights_2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto weights_3 = create_constant_with_zeros(weights_shape3, {{2, 3, 4, 5, 6, 7}, {}, {}, {}});
    auto conv3 = std::make_shared<opset5::Convolution>(input, weights_3, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto concat = std::make_shared<opset5::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

    auto add_const = create_constant_with_zeros(Shape{1, 32, 1, 1}, {{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}});
    auto add = std::make_shared<opset5::Add>(concat, add_const);

    auto f = std::make_shared<Function>(NodeVector{add}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::InitMasks>();
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights_1.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_2.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_3.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(add->output(0)),  Mask({{}, {}, {}, {}}));


    compare_masks(*getMask(concat->output(0)),  Mask({{}, {}, {}, {}}));
}
