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

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2, 3}, {}, {}});
    auto sub = std::make_shared<opset5::Subtract>(relu, sub_const);

    auto mul_const = create_constant_with_zeros(Shape{6, 1, 1}, {{2}, {}, {}});
    auto mul = std::make_shared<opset5::Subtract>(sub, mul_const);

    auto weights2 = opset5::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(mul, weights2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
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

    auto mul_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2}, {}, {}});
    auto mul = std::make_shared<opset5::Subtract>(sub, mul_const);

    auto weights2 = opset5::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(mul, weights2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto f = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    pass::Manager m;
    m.register_pass<pass::PropagateMasks>();
    m.run_passes(f);

    compare_masks(*getMask(weights->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),     Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{}, {}, {}}));
    compare_masks(*getMask(mul_const), Mask({{}, {}, {}}));
    compare_masks(*getMask(weights2->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),    Mask({{}, {}, {}, {}}));
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