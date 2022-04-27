// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <pruning.hpp>
#include <mask_attribute.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/coordinate_transform.hpp>
#include <ngraph/pass/manager.hpp>
#include <inference_engine.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

#define VISUALIZE_TESTS_TREE false
#define VISUALIZE_TREE_ROOT "/tmp/"

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

            NGRAPH_SUPPRESS_DEPRECATED_START
            CoordinateTransform iter(shape, coord_begin, coord_end);
            for (const Coordinate & coord : iter) {
                values[iter.index(coord)] = 0;
            }
            NGRAPH_SUPPRESS_DEPRECATED_END
        }
    }
    return std::make_shared<opset5::Constant>(element::f32, shape, values);
}

// Uncomment, specify PRUNING_TARGET_IR_PATH var and
// include <openvino/util/env_util.hpp> to check pruning on given IR
//TEST(TransformationTests, PruneIRTest) {
//    InferenceEngine::Core core;
//
//    const std::string input_model = ov::util::getenv_string("PRUNING_TARGET_IR_PATH");
//    if (input_model == "")
//        return;
//
//    auto function = core.ReadNetwork(input_model).getFunction();
//
//    pass::Manager m;
//    m.register_pass<pass::InitMasks>();
//    m.register_pass<pass::PropagateMasks>();
//
//    // VisualizeTree modifier helps to print Masks and mark nodes with masks
//    auto modifier = [](const Node& node, std::vector<std::string>& attributes) {
//        std::stringstream ss;
//        size_t index{0};
//        for (const auto & output : node.outputs()) {
//            if (const auto & mask = getMask(output)) {
//                if (!mask->all_dims_are_empty()) {
//                    attributes.emplace_back("color=green");
//                    attributes.emplace_back("penwidth=2");
//                }
//                ss << "Mask(" << index << ") : " << *mask << "\\n";
//            }
//            index++;
//        }
//        if (!ss.str().empty()) {
//            auto label = std::find_if(attributes.begin(), attributes.end(),
//                                   [](const std::string & value) { return value.find("label=") != std::string::npos; });
//            if (label != attributes.end()) {
//                label->pop_back();
//                *label += "\n" + ss.str() + "\"";
//            } else {
//                attributes.push_back("label=\"" + ss.str() + "\"");
//            }
//        }
//    };
//
//    m.register_pass<ngraph::pass::VisualizeTree>(std::string(VISUALIZE_TREE_ROOT) + "PruneIRTest_with_masks.svg", modifier);
//    m.register_pass<pass::ShrinkWeights>();
//
//    if (VISUALIZE_TESTS_TREE)
//        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneIRTest.svg").run_on_function(function);
//    m.run_passes(function);
//}


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
    NGRAPH_SUPPRESS_DEPRECATED_START
    CoordinateTransform iter(weights_shape, {0, 1, 0, 0}, {6, 2, 3, 3});
    for (const Coordinate & coord : iter) {
        values[iter.index(coord)] = 0;
    }
    NGRAPH_SUPPRESS_DEPRECATED_END

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


TEST_F(TransformationTestsF, PropagateMasksBasic) {
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

    auto sub_const = create_constant_with_zeros(Shape{6, 1, 1}, {{1, 2}, {}, {}});
    auto sub = std::make_shared<opset5::Subtract>(add, sub_const);

    auto mul_const = create_constant_with_zeros(Shape{1, 6, 1, 1}, {{}, {3}, {}, {}});
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, mul_const);

    auto weights2 = create_constant_with_zeros(weights_shape2, {{1, 2}, {1, 2, 3}, {}, {}});
    auto conv2 = std::make_shared<opset5::Convolution>(mul, weights2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});

    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);

        auto weights = opset5::Constant::create(element::f32, {weights_shape[0] - 3, weights_shape[1], weights_shape[2] , weights_shape[3]}, {0});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        auto relu = std::make_shared<opset5::Relu>(conv);

        auto add_const = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1});
        auto add = std::make_shared<opset5::Add>(relu, add_const);

        auto sub_const  = opset5::Constant::create(element::f32, Shape{3, 1, 1}, {1});
        auto sub = std::make_shared<opset5::Subtract>(add, sub_const);

        auto mul_const = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1});
        auto mul = std::make_shared<ov::op::v1::Multiply>(sub, mul_const);

        auto weights2 = opset5::Constant::create(element::f32, {weights_shape2[0], weights_shape2[1] - 3,  weights_shape2[2], weights_shape2[3]}, {1});
        auto conv2 = std::make_shared<opset5::Convolution>(mul, weights2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        function_ref = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksBasic.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add_const), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{1, 2, 3}, {}, {}}));
    compare_masks(*getMask(mul_const), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(weights2.get_node_shared_ptr()->output(0)), Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),    Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, PropagateMasksDynamicConvolution) {
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
    function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights = opset5::Constant::create(element::f32, {weights_shape[0] - 1, weights_shape[1], weights_shape[2], weights_shape[3]}, {0});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        auto relu = std::make_shared<opset5::Relu>(conv);

        auto sub_const = create_constant_with_zeros(Shape{5, 1, 1}, {{}, {}, {}});
        auto sub = std::make_shared<opset5::Subtract>(relu, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{5, 1, 1}, {{2}, {}, {}});
        auto mul = std::make_shared<opset5::Subtract>(sub, mul_const);

        auto weights2 = opset5::Constant::create(element::f32, {weights_shape2[0], weights_shape2[1] - 1, weights_shape2[2], weights_shape2[3]}, {0});
        auto conv2 = std::make_shared<opset5::Convolution>(mul, weights2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        function_ref = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksDynamicConvolution.svg").run_on_function(function);

    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }

    compare_masks(*getMask(weights->output(0)),  Mask({{2}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),     Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(sub_const), Mask({{2}, {}, {}}));
    compare_masks(*getMask(mul_const), Mask({{2}, {}, {}}));
    compare_masks(*getMask(weights2->output(0)), Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),    Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST(TransformationTests, PropagateMasksDynamicReshape) {
    PartialShape input_shape{Dimension::dynamic(), 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_shape2{6, 6, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = opset5::Constant::create(element::f32, weights_shape, {0});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto reshape = std::make_shared<opset5::Reshape>(relu, opset5::Constant::create(element::i64, Shape{4}, {-1, 6, 64, 64}), true);

    auto weights2 = opset5::Constant::create(element::f32, weights_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(reshape, weights2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksDynamicReshape.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),     Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(reshape), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(weights2->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),     Mask({{}, {}, {}, {}}));
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

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksDynamicGroupConvolution.svg").run_on_function(f);

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

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksEmpty.svg").run_on_function(f);

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


TEST_F(TransformationTestsF, PropagateMaskPassThrough) {
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
    function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights_const_1 = create_constant_with_zeros({weights_shape[0] - 3, weights_shape[1], weights_shape[2], weights_shape[3]}  , {{}, {}, {}, {}});
        weights_const_1.get_node_shared_ptr()->set_friendly_name("weights_1");

        auto conv_1 = std::make_shared<opset5::Convolution>(input, weights_const_1, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        // Adding a couple of PassThrough operations
        auto relu = std::make_shared<opset5::Relu>(conv_1);

        auto clamp = std::make_shared<opset5::Clamp>(relu, 0, 6);

        auto pads_begin = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = opset5::Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = std::make_shared<opset5::Pad>(clamp, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto max_pool = std::make_shared<opset5::MaxPool>(pad, Strides{1, 1},
                                                          Shape{0, 0}, Shape{1, 1}, Shape{4, 4});

        auto weights2 = opset5::Constant::create(element::f32, {weight_shape2[0], weight_shape2[1] - 3, weight_shape2[2], weight_shape2[3]}, {0});
        auto conv2 = std::make_shared<opset5::Convolution>(max_pool, weights2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        function_ref = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMaskPassThrough.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights_const_1.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(relu->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(clamp->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(max_pool->output(0)),     Mask({{}, {1, 2, 3}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, PropagateMasksHardDependencies) {
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

    auto matmul_const = opset5::Constant::create(element::f32, Shape{6, 100}, {1.});
    auto matmul = std::make_shared<opset5::MatMul>(reshape, matmul_const);
    matmul->set_friendly_name("matmul");

    auto add2 = std::make_shared<opset5::Add>(conv2, create_constant_with_zeros({6, 1, 1}, {{2}, {}, {}}));
    add2->set_friendly_name("add2");

    Shape weights_shape3{6, 6, 1, 1};
    auto weights3 = opset5::Constant::create(element::f32, weights_shape3, {0});
    weights3->set_friendly_name("weights3");

    auto conv3 = std::make_shared<opset5::Convolution>(add2, weights3, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    conv3->set_friendly_name("conv3");

    function = std::make_shared<Function>(NodeVector{matmul, conv3}, ParameterVector{input1, input2});
    {
        auto input1 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        input1->set_friendly_name("input1");

        Shape weights1_shape{6, 3, 3, 3};
        auto weights1 = create_constant_with_zeros({
                                                       weights1_shape[0] - 1,
                                                       weights1_shape[1],
                                                       weights1_shape[2],
                                                       weights1_shape[3]
                                                   }, {{}, {}, {}, {}});
        weights1.get_node_shared_ptr()->set_friendly_name("weights1");

        auto conv1 = std::make_shared<opset5::Convolution>(input1, weights1, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        conv1->set_friendly_name("conv1");

        auto relu = std::make_shared<opset5::Relu>(conv1);
        relu->set_friendly_name("relu");

        auto input2 = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        input2->set_friendly_name("input2");

        Shape weights2_shape{6, 3, 3, 3};
        auto weights2 = create_constant_with_zeros({weights2_shape[0] - 1,
                                                    weights2_shape[1],
                                                    weights2_shape[2],
                                                    weights2_shape[3]
                                                   }, {{2, 3}, {}, {}, {}});
        weights2.get_node_shared_ptr()->set_friendly_name("weights2");

        auto conv2 = std::make_shared<opset5::Convolution>(input2, weights2, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        conv2->set_friendly_name("conv2");

        auto add1 = std::make_shared<opset5::Add>(conv2, conv1);
        add1->set_friendly_name("add1");

        auto reshape = std::make_shared<opset5::Reshape>(add1, opset5::Constant::create(element::i64, Shape{2}, {1, 5}), true);
        reshape->set_friendly_name("reshape");

        auto matmul = std::make_shared<opset5::MatMul>(reshape, opset5::Constant::create(element::f32, Shape{5, 100}, {1.}));
        matmul->set_friendly_name("matmul");

        auto add2 = std::make_shared<opset5::Add>(conv2, create_constant_with_zeros({5, 1, 1}, {{}, {}, {}}));
        add2->set_friendly_name("add2");

        Shape weights_shape3{6, 6, 1, 1};
        auto weights3 = opset5::Constant::create(element::f32,
                                                   {weights_shape3[0],
                                                    weights_shape3[1] - 1,
                                                    weights_shape3[2],
                                                    weights_shape3[3]
                                                   }, {0});
        weights3->set_friendly_name("weights3");

        auto conv3 = std::make_shared<opset5::Convolution>(add2, weights3, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        conv3->set_friendly_name("conv3");

        function_ref = std::make_shared<Function>(NodeVector{matmul, conv3}, ParameterVector{input1, input2});
    }

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksHardDependencies.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights1.get_node_shared_ptr()->output(0)), Mask({{2}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {2}, {}, {}}));

    compare_masks(*getMask(weights2.get_node_shared_ptr()->output(0)), Mask({{2}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {2}, {}, {}}));

    compare_masks(*getMask(weights3->output(0)), Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(add1->output(0)),  Mask({{}, {2}, {}, {}}));
    compare_masks(*getMask(add2->output(0)),  Mask({{}, {2}, {}, {}}));

    compare_masks(*getMask(matmul_const->output(0)), Mask({{2}, {}}));
    compare_masks(*getMask(matmul->output(0)),  Mask({{}, {}}));

    // TODO: add checks after MatMul/Reshape/Pooling mask propagation is ready
    //compare_masks(*getMask(weights),  Mask({{0, 1, 2, 3, 4, 5}, {}, {}, {}}));
    //compare_masks(*getMask(conv),     Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
    //compare_masks(*getMask(relu),     Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
    //compare_masks(*getMask(weights2), Mask({{}, {0, 1, 2, 3, 4, 5}, {}, {}}));
    //compare_masks(*getMask(conv2),    Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, PropagateMasksQuantizedGroupConvolution) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weights_group_shape{8, 1, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");

    auto weights1 = create_constant_with_zeros(weights_shape, {{0, 1, 2, 3, 4}, {}, {}, {}});
    auto conv1 = std::make_shared<opset5::Convolution>(input, weights1, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto weights_group = opset5::Constant::create(element::i8, weights_group_shape, {0});
    weights_group->set_friendly_name("weights_group");

    auto convert = std::make_shared<opset5::Convert>(weights_group, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});

    auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{}, {}, {}, {}});
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
    function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);

        auto weights1 = create_constant_with_zeros({weights_shape[0] - 5, weights_shape[1], weights_shape[2], weights_shape[3]}, {{}, {}, {}, {}});
        auto conv1 = std::make_shared<opset5::Convolution>(input, weights1, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        auto weights_group = opset5::Constant::create(element::i8,
                                                      {
                                                          weights_group_shape[0] - 5,
                                                          weights_group_shape[1],
                                                          weights_group_shape[2],
                                                          weights_group_shape[3]
                                                      }, {0});

        auto convert = std::make_shared<opset5::Convert>(weights_group, element::f32);

        auto sub_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});

        auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});
        auto mul = std::make_shared<opset5::Multiply>(sub, mul_const);


        auto reshape_const = opset5::Constant::create(element::i64, Shape{5}, {8, 1, 1, 3, 3});

        const auto axis = opset6::Constant::create(ov::element::i8, {}, {0});
        auto dims_to_keep_vec = std::vector<size_t>{2, 3, 4};
        const auto dims_to_keep = opset6::Constant::create(reshape_const->get_element_type(), {dims_to_keep_vec.size()}, dims_to_keep_vec);
        const auto reshape_gather = std::make_shared<opset6::Gather>(reshape_const, dims_to_keep, axis);
        const auto reshape_concat = std::make_shared<opset6::Concat>(
            NodeVector{opset6::Constant::create(reshape_const->get_element_type(), {2}, {-1, 1}), reshape_gather}, 0);
        auto reshape = std::make_shared<opset5::Reshape>(mul, reshape_concat, false);

        auto conv_group = std::make_shared<opset5::GroupConvolution>(conv1, reshape, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 3, 1, 1}, {{}, {}, {}, {}});;
        auto add = std::make_shared<opset5::Add>(conv_group, add_const);

        auto weights_2 = opset5::Constant::create(element::f32, {weight_shape2[0], weight_shape2[1] - 5, weight_shape2[2], weight_shape2[3]}, {0});
        auto conv2 = std::make_shared<opset5::Convolution>(add, weights_2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        function_ref = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksQuantizedGroupConvolution.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }

    compare_masks(*getMask(weights1.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_group->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(reshape->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}, {}}));

    compare_masks(*getMask(conv_group->output(0)),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, PropagateMasksQuantizedGroupConvolutionWithShapeOf) {
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

    auto shape_of = std::make_shared<opset6::ShapeOf>(mul);
    auto axis = opset6::Constant::create(ov::element::i8, {}, {0});
    auto split_lenghts = opset6::Constant::create(ov::element::i8, {2}, {1, -1});
    auto variadic_split = std::make_shared<opset6::VariadicSplit>(shape_of, axis, split_lenghts);
    auto div_const = opset6::Constant::create(ov::element::i64, {1}, {8});
    auto div = std::make_shared<opset6::Divide>(variadic_split->output(0), div_const);
    auto reshape_concat = std::make_shared<opset6::Concat>(
        OutputVector{opset6::Constant::create(shape_of->get_element_type(), {1}, {8})->output(0),
                  div->output(0), variadic_split->output(1)}, 0);

    auto reshape = std::make_shared<opset5::Reshape>(mul, reshape_concat, false);

    auto conv_group = std::make_shared<opset5::GroupConvolution>(conv1, reshape, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto add_const = create_constant_with_zeros(Shape{1, 8, 1, 1}, {{}, {0, 1, 2, 3, 4}, {}, {}});;
    auto add = std::make_shared<opset5::Add>(conv_group, add_const);
    add->set_friendly_name("add");

    auto weights_2 = opset5::Constant::create(element::f32, weight_shape2, {0});
    auto conv2 = std::make_shared<opset5::Convolution>(add, weights_2, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);

        auto weights1 = create_constant_with_zeros({weights_shape[0] - 4, weights_shape[1], weights_shape[2], weights_shape[3]}, {{}, {}, {}, {}});
        auto conv1 = std::make_shared<opset5::Convolution>(input, weights1, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        auto weights_group = opset5::Constant::create(element::i8,
                                                      {
                                                          weights_group_shape[0] - 4,
                                                          weights_group_shape[1],
                                                          weights_group_shape[2],
                                                          weights_group_shape[3]
                                                      }, {0});

        auto convert = std::make_shared<opset5::Convert>(weights_group, element::f32);

        auto sub_const = create_constant_with_zeros(Shape{4, 1, 1, 1}, {{}, {}, {}, {}});

        auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{4, 1, 1, 1}, {{}, {}, {}, {}});
        auto mul = std::make_shared<opset5::Multiply>(sub, mul_const);


        auto shape_of = std::make_shared<opset6::ShapeOf>(mul);
        auto axis = opset6::Constant::create(ov::element::i8, {}, {0});
        auto split_lenghts = opset6::Constant::create(ov::element::i8, {2}, {1, -1});
        auto variadic_split = std::make_shared<opset6::VariadicSplit>(shape_of, axis, split_lenghts);
        auto div_const = opset6::Constant::create(ov::element::i64, {1}, {8});
        auto div = std::make_shared<opset6::Divide>(variadic_split->output(0), div_const);

        auto reshape_concat = std::make_shared<opset6::Concat>(
            OutputVector{opset6::Constant::create(shape_of->get_element_type(), {1}, {1})->output(0),
                         div->output(0), variadic_split->output(1)}, 0);

        const auto axis_1 = opset6::Constant::create(ov::element::i8, {}, {0});
        auto dims_to_keep_vec = std::vector<size_t>{2, 3, 4};
        const auto dims_to_keep = opset6::Constant::create(reshape_concat->get_element_type(), {dims_to_keep_vec.size()}, dims_to_keep_vec);
        const auto new_reshape_gather = std::make_shared<opset6::Gather>(reshape_concat, dims_to_keep, axis_1);
        const auto new_reshape_concat = std::make_shared<opset6::Concat>(
            NodeVector{opset6::Constant::create(reshape_concat->get_element_type(), {2}, {-1, 1}), new_reshape_gather}, 0);
        auto reshape = std::make_shared<opset5::Reshape>(mul, new_reshape_concat, false);

        auto conv_group = std::make_shared<opset5::GroupConvolution>(conv1, reshape, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 4, 1, 1}, {{}, {}, {}, {}});;
        auto add = std::make_shared<opset5::Add>(conv_group, add_const);

        auto weights_2 = opset5::Constant::create(element::f32, {weight_shape2[0], weight_shape2[1] - 4, weight_shape2[2], weight_shape2[3]}, {0});
        auto conv2 = std::make_shared<opset5::Convolution>(add, weights_2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        function_ref = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksQuantizedGroupConvolutionWithShapeOf.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }

    compare_masks(*getMask(weights1.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0 , 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_group->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}}));

    compare_masks(*getMask(reshape->output(0)), Mask({{0, 1, 2, 3}, {}, {}, {}, {}}));

    compare_masks(*getMask(conv_group->output(0)),  Mask({{}, {0, 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)),  Mask({{}, {0, 1, 2, 3}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, PropagateMasksFakeQuantizePerTensor) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_1 = opset5::Constant::create(element::i8, weights_shape, {0});
    weights_1->set_friendly_name("weights_int8_const");

    auto convert = std::make_shared<opset5::Convert>(weights_1, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});

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
    function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights_1 = opset5::Constant::create(element::i8, {
                                                                    weights_shape[0] - 5,
                                                                    weights_shape[1],
                                                                    weights_shape[2],
                                                                    weights_shape[3],
                                                                }, {0});

        auto convert = std::make_shared<opset5::Convert>(weights_1, element::f32);

        auto sub_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});

        auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});
        auto mul = std::make_shared<opset5::Multiply>(sub, mul_const);

        auto conv1 = std::make_shared<opset5::Convolution>(input, mul, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 3, 1, 1}, {{}, {}, {}, {}});;
        auto add = std::make_shared<opset5::Add>(conv1, add_const);

        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 1, 1, 1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {1});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

        auto weights_2 = opset5::Constant::create(element::f32, {
                                                                    weight_shape2[0],
                                                                    weight_shape2[1] - 5,
                                                                    weight_shape2[2],
                                                                    weight_shape2[3],
                                                                }, {0});
        auto conv2 = std::make_shared<opset5::Convolution>(fq, weights_2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        function_ref  = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksFakeQuantizePerTensor.svg").run_on_function(function);

    {
        pass::Manager m;
        // Masks for fq input parammeters didn't saved after
        // ShrinkWeights pass so pruning transformation is splitted
        // on propagation and shrinking passes
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }    pass::Manager m;

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
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST(TransformationTests, PropagateMasksFakeQuantizePerTensor1DScale) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_1 = opset5::Constant::create(element::i8, weights_shape, {0});
    weights_1->set_friendly_name("weights_int8_const");

    auto convert = std::make_shared<opset5::Convert>(weights_1, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{1}, {{}});

    auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{1}, {{}});
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
    auto function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksFakeQuantizePerTensor1DScale.svg").run_on_function(function);

    {
        pass::Manager m;
        m.register_pass<pass::Pruning>();
        m.run_passes(function);
    }

    compare_masks(*getMask(weights_1->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(add->output(0)),  Mask({{}, {},  {}, {}}));

    compare_masks(*getMask(fq->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST_F(TransformationTestsF, PropagateMasksFakeQuantizePerChannel) {
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{8, 3, 3, 3};
    Shape weight_shape2{3, 8, 3, 3};
    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    input->set_friendly_name("input");
    auto weights_1 = opset5::Constant::create(element::i8, weights_shape, {0});
    weights_1->set_friendly_name("weights_int8_const");

    auto convert = std::make_shared<opset5::Convert>(weights_1, element::f32);
    convert->set_friendly_name("convert");

    auto sub_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{0, 1, 2, 3, 4}, {}, {}, {}});

    auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);
    sub->set_friendly_name("sub");

    auto mul_const = create_constant_with_zeros(Shape{8, 1, 1, 1}, {{}, {}, {}, {}});
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
    function = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights_1 = opset5::Constant::create(element::i8, {
                                                                  weights_shape[0] - 5,
                                                                  weights_shape[1],
                                                                  weights_shape[2],
                                                                  weights_shape[3]
                                                                }, {0});

        auto convert = std::make_shared<opset5::Convert>(weights_1, element::f32);

        auto sub_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});

        auto sub = std::make_shared<opset5::Subtract>(convert, sub_const);

        auto mul_const = create_constant_with_zeros(Shape{3, 1, 1, 1}, {{}, {}, {}, {}});
        auto mul = std::make_shared<opset5::Multiply>(sub, mul_const);

        auto conv1 = std::make_shared<opset5::Convolution>(input, mul, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 3, 1, 1}, {{}, {}, {}, {}});;
        auto add = std::make_shared<opset5::Add>(conv1, add_const);

        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1});
        auto output_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 8);

        auto weights_2 = opset5::Constant::create(element::f32, {
                                                                    weight_shape2[0],
                                                                    weight_shape2[1] - 5,
                                                                    weight_shape2[2],
                                                                    weight_shape2[3]
                                                                } , {0});
        auto conv2 = std::make_shared<opset5::Convolution>(fq, weights_2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        function_ref  = std::make_shared<Function>(NodeVector{conv2}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksFakeQuantizePerChannel.svg").run_on_function(function);
    {
        pass::Manager m;
        // Masks for fq input parammeters didn't saved after
        // ShrinkWeights pass so pruning transformation is splitted
        // on propagation and shrinking passes
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights_1->output(0)), Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub_const.get_node_shared_ptr()->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(sub->output(0)),  Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(mul_const.get_node_shared_ptr()->output(0)),  Mask({{0 , 1, 2, 3, 4}, {}, {}, {}}));
    compare_masks(*getMask(mul->output(0)),  Mask({{0, 1, 2, 3, 4}, {}, {}, {}}));

    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)),  Mask({{}, {0 , 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(add->output(0)),  Mask({{}, {0, 1, 2, 3, 4},  {}, {}}));

    compare_masks(*getMask(fq->output(0)),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));

    compare_masks(*getMask(weights_2->output(0)),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(fq->input(1).get_source_output()),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(2).get_source_output()),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(3).get_source_output()),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    compare_masks(*getMask(fq->input(4).get_source_output()),  Mask({{}, {0, 1, 2, 3, 4}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, TestConcatMaskPropagation) {
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
    function = std::make_shared<Function>(NodeVector{conv_out}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights_1 = create_constant_with_zeros({
                                                        weights_shape1[0] - 4,
                                                        weights_shape1[1],
                                                        weights_shape1[2],
                                                        weights_shape1[3]
                                                    }, {{}, {}, {}, {}});
        auto conv1 = std::make_shared<opset5::Convolution>(input, weights_1, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto weights_2 = create_constant_with_zeros({
                                                        weights_shape2[0] - 4,
                                                        weights_shape2[1],
                                                        weights_shape2[2],
                                                        weights_shape2[3],
                                                    }, {{}, {}, {}, {}});
        auto conv2 = std::make_shared<opset5::Convolution>(input, weights_2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto weights_3 = create_constant_with_zeros({
                                                        weights_shape3[0] - 4,
                                                        weights_shape3[1],
                                                        weights_shape3[2],
                                                        weights_shape3[3],
                                                    }, {{}, {}, {}, {}});
        auto conv3 = std::make_shared<opset5::Convolution>(input, weights_3, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto concat = std::make_shared<opset5::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

        auto weights_out_conv = create_constant_with_zeros({
                                                                weight_shape_out_conv[0],
                                                                weight_shape_out_conv[1] - 12,
                                                                weight_shape_out_conv[2],
                                                                weight_shape_out_conv[3],
                                                            }, {{}, {}, {}, {}});
        auto conv_out = std::make_shared<opset5::Convolution>(concat, weights_out_conv, Strides(2, 1),
                                                       CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        function_ref = std::make_shared<Function>(NodeVector{conv_out}, ParameterVector{input});
    }

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "TestConcatMaskPropagation.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights_1.get_node_shared_ptr()->output(0)),  Mask({{0, 1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv1->output(0)),  Mask({{}, {0, 1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_2.get_node_shared_ptr()->output(0)),  Mask({{7, 8, 9, 10}, {}, {}, {}}));
    compare_masks(*getMask(conv2->output(0)),  Mask({{}, {7, 8, 9, 10}, {}, {}}));

    compare_masks(*getMask(weights_3.get_node_shared_ptr()->output(0)),  Mask({{4, 5, 6, 7}, {}, {}, {}}));
    compare_masks(*getMask(conv3->output(0)),  Mask({{}, {4, 5, 6, 7}, {}, {}}));

    compare_masks(*getMask(concat->output(0)),  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
    compare_masks(*getMask(weights_out_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {0, 1, 2, 3, 15, 16, 17, 18, 28, 29, 30, 31}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, TestConcatMaskPropagationUp) {
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
    function = std::make_shared<Function>(NodeVector{conv_out}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights_1 = create_constant_with_zeros({
                                                        weights_shape1[0] - 4,
                                                        weights_shape1[1],
                                                        weights_shape1[2],
                                                        weights_shape1[3],
                                                    }, {{}, {}, {}, {}});
        auto conv1 = std::make_shared<opset5::Convolution>(input, weights_1, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto weights_2 = create_constant_with_zeros({
                                                        weights_shape2[0] - 4,
                                                        weights_shape2[1],
                                                        weights_shape2[2],
                                                        weights_shape2[3],
                                                    }, {{}, {}, {}, {}});
        auto conv2 = std::make_shared<opset5::Convolution>(input, weights_2, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto weights_3 = create_constant_with_zeros({
                                                        weights_shape3[0] - 4,
                                                        weights_shape3[1],
                                                        weights_shape3[2],
                                                        weights_shape3[3],
                                                    }, {{}, {}, {}, {}});
        auto conv3 = std::make_shared<opset5::Convolution>(input, weights_3, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        auto concat = std::make_shared<opset5::Concat>(OutputVector{conv1->output(0), conv2->output(0), conv3->output(0)}, 1);

        auto add_const = create_constant_with_zeros(Shape{1, 20, 1, 1}, {{}, {}, {}, {}});
        auto add = std::make_shared<opset5::Add>(concat, add_const);

        auto weights_out_conv = create_constant_with_zeros({
                                                        weight_shape_out_conv[0],
                                                        weight_shape_out_conv[1] - 12,
                                                        weight_shape_out_conv[2],
                                                        weight_shape_out_conv[3],
                                                    }, {{}, {}, {}, {}});
        auto conv_out = std::make_shared<opset5::Convolution>(add, weights_out_conv, Strides(2, 1),
                                                              CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

        function_ref = std::make_shared<Function>(NodeVector{conv_out}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "TestConcatMaskPropagationUp.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
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
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "TestConcatMaskPropagationUpEmpty.svg").run_on_function(f);

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


TEST_F(TransformationTestsF, PruneConvIsClosingAndInGroup) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                      CoordinateDiff(2, 0),
                                                                      CoordinateDiff(2, 0),
                                                                      Strides(2, 1));

    auto add_const = create_constant_with_zeros(Shape{1, 6, 1, 1}, {{}, {1, 2, 3, 4, 5}, {}, {}});
    auto add = std::make_shared<opset5::Add>(conv, add_const);

    auto conv_1_shape = Shape{weightsShape[0], weightsShape[0], 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(add, conv_1_weights, Strides(2, 1),
                                                                                 CoordinateDiff(2, 0),
                                                                                 CoordinateDiff(2, 0),
                                                                                 Strides(2, 1));

    auto add_1 = std::make_shared<opset5::Add>(conv_1, conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset5::Convolution>(add_1, weights_end_conv, Strides(2, 1),
                                                                                 CoordinateDiff(2, 0),
                                                                                 CoordinateDiff(2, 0),
                                                                                 Strides(2, 1));

    function = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneConvIsClosingAndInGroup.svg").run_on_function(function);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros({
                                                        weightsShape[0] - 3,
                                                        weightsShape[1],
                                                        weightsShape[2],
                                                        weightsShape[3],
                                                   }, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                          CoordinateDiff(2, 0),
                                                                          CoordinateDiff(2, 0),
                                                                          Strides(2, 1));

        auto add_const = create_constant_with_zeros(Shape{1, 3, 1, 1}, {{}, {}, {}, {}});
        auto add = std::make_shared<opset5::Add>(conv, add_const);

        auto conv_1_shape = Shape{weightsShape[0] - 3, weightsShape[0] - 3, 1, 1};
        auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset5::Convolution>(add, conv_1_weights, Strides(2, 1),
                                                                                     CoordinateDiff(2, 0),
                                                                                     CoordinateDiff(2, 0),
                                                                                     Strides(2, 1));


        auto add_1 = std::make_shared<opset5::Add>(conv_1, conv);

        auto end_conv_shape = Shape{weightsShape[1], weightsShape[0] - 3, 1, 1};
        auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{}, {}, {}, {}});
        auto end_conv = std::make_shared<opset5::Convolution>(add_1, weights_end_conv, Strides(2, 1),
                                                                                     CoordinateDiff(2, 0),
                                                                                     CoordinateDiff(2, 0),
                                                                                     Strides(2, 1));
        function_ref = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input});
    }
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(add_const.get_node_shared_ptr()->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add_1->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)),  Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST(TransformationTests, PruneBranchingStopOp) {
    // Checks case of branching with stop op
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0),
                                                      CoordinateDiff(2, 0),
                                                      Strides(2, 1));
    // Branching stop op
    Shape group_conv_weights_shape{3, 2, 2, 1, 1};
    auto group_conv_weights = opset5::Constant::create(element::f32, group_conv_weights_shape, {0});
    auto group_conv = std::make_shared<opset5::GroupConvolution>(conv, group_conv_weights, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto conv_1_shape = Shape{weightsShape[0], 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(group_conv, conv_1_weights, Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    // Multiply will try to propagate a non zero masks of the conv_1 up
    // and the mask should be invalidated by group conv stop op mask
    auto mul = std::make_shared<opset5::Multiply>(conv_1, conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset5::Convolution>(mul, weights_end_conv, Strides(2, 1),
                                                          CoordinateDiff(2, 0),
                                                          CoordinateDiff(2, 0),
                                                          Strides(2, 1));

    auto function = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input}, "RestrictedReduceMeanBranching");

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneBranchingStopOp.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST(TransformationTests, PruneStopOpUp) {
    // Checks case of branching with stop op
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0),
                                                      CoordinateDiff(2, 0),
                                                      Strides(2, 1));
    // Branching stop op
    Shape group_conv_weights_shape{3, 2, 2, 1, 1};
    auto group_conv_weights = opset5::Constant::create(element::f32, group_conv_weights_shape, {0});
    auto group_conv = std::make_shared<opset5::GroupConvolution>(conv, group_conv_weights, Strides(2, 1),
                                                           CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));

    auto conv_1_shape = Shape{weightsShape[0], 6, 1, 1};

    auto mul_const = create_constant_with_zeros(Shape{1, 6, 16, 16}, {{}, {1, 2, 3}, {}, {}});
    auto mul = std::make_shared<opset5::Multiply>(group_conv, mul_const);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset5::Convolution>(mul, weights_end_conv, Strides(2, 1),
                                                          CoordinateDiff(2, 0),
                                                          CoordinateDiff(2, 0),
                                                          Strides(2, 1));
    auto function = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input}, "StopOpUp");

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneStopOpUp.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST_F(TransformationTestsF, PruneReducelayerUp) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                      CoordinateDiff(2, 0),
                                                                      CoordinateDiff(2, 0),
                                                                      Strides(2, 1));

    auto reduce_const = opset5::Constant::create(element::i64, Shape{2}, {2, 3});
    auto reduce_mean = std::make_shared<opset5::ReduceMean>(conv, reduce_const, true);

    auto conv_1_shape = Shape{12, 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(reduce_mean, conv_1_weights, Strides(2, 1),
                                                                                 CoordinateDiff(2, 0),
                                                                                 CoordinateDiff(2, 0),
                                                                                 Strides(2, 1));

    function = std::make_shared<ngraph::Function>(OutputVector{conv_1}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros({
                                                    weightsShape[0] - 3,
                                                    weightsShape[1],
                                                    weightsShape[2],
                                                    weightsShape[3]
                                                  }, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                          CoordinateDiff(2, 0),
                                                                          CoordinateDiff(2, 0),
                                                                          Strides(2, 1));

        auto reduce_const = opset5::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce_mean = std::make_shared<opset5::ReduceMean>(conv, reduce_const, true);

        auto conv_1_shape = Shape{12, 3, 1, 1};
        auto conv_1_weights = create_constant_with_zeros({
                                                            conv_1_shape[0],
                                                            conv_1_shape[1],
                                                            conv_1_shape[2],
                                                            conv_1_shape[3]
                                                         }, {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset5::Convolution>(reduce_mean, conv_1_weights, Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
        function_ref = std::make_shared<ngraph::Function>(OutputVector{conv_1}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneReducelayerUp.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST_F(TransformationTestsF, PruneReduceLayerDown) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                      CoordinateDiff(2, 0),
                                                                      CoordinateDiff(2, 0),
                                                                      Strides(2, 1));

    auto reduce_const = opset5::Constant::create(element::i64, Shape{2}, {2, 3});
    auto reduce_mean = std::make_shared<opset5::ReduceMean>(conv, reduce_const, true);

    auto conv_1_shape = Shape{weightsShape[0], weightsShape[0], 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(reduce_mean, conv_1_weights, Strides(2, 1),
                                                                                 CoordinateDiff(2, 0),
                                                                                 CoordinateDiff(2, 0),
                                                                                 Strides(2, 1));

    auto add_1 = std::make_shared<opset5::Add>(conv_1, conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset5::Convolution>(add_1, weights_end_conv, Strides(2, 1),
                                                                                 CoordinateDiff(2, 0),
                                                                                 CoordinateDiff(2, 0),
                                                                                 Strides(2, 1));

    function = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros({
                                                        weightsShape[0] - 3,
                                                        weightsShape[1],
                                                        weightsShape[2],
                                                        weightsShape[3],
                                                   }, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                          CoordinateDiff(2, 0),
                                                                          CoordinateDiff(2, 0),
                                                                          Strides(2, 1));

        auto reduce_const = opset5::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce_mean = std::make_shared<opset5::ReduceMean>(conv, reduce_const, true);

        auto conv_1_shape = Shape{weightsShape[0] - 3, weightsShape[0] - 3, 1, 1};
        auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset5::Convolution>(reduce_mean, conv_1_weights, Strides(2, 1),
                                                                                     CoordinateDiff(2, 0),
                                                                                     CoordinateDiff(2, 0),
                                                                                     Strides(2, 1));


        auto add_1 = std::make_shared<opset5::Add>(conv_1, conv);

        auto end_conv_shape = Shape{weightsShape[1], weightsShape[0] - 3, 1, 1};
        auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{}, {}, {}, {}});
        auto end_conv = std::make_shared<opset5::Convolution>(add_1, weights_end_conv, Strides(2, 1),
                                                                                     CoordinateDiff(2, 0),
                                                                                     CoordinateDiff(2, 0),
                                                                                     Strides(2, 1));
        function_ref = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneReduceLayerDown.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(reduce_mean->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(add_1->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)),  Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST(TransformationTests, PruneStopReducelayerUp) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0),
                                                      CoordinateDiff(2, 0),
                                                      Strides(2, 1));

    auto reduce_const = opset5::Constant::create(element::i64, Shape{3}, {1, 2, 3});
    auto reduce_mean = std::make_shared<opset5::ReduceMean>(conv, reduce_const, true);

    auto conv_1_shape = Shape{12, 1, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(reduce_mean, conv_1_weights, Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto function = std::make_shared<ngraph::Function>(OutputVector{conv_1}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneStopReducelayerUp.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST(TransformationTests, PruneStopReduceLayerDown) {
    // Checks case of branching with stop op
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0),
                                                      CoordinateDiff(2, 0),
                                                      Strides(2, 1));
    // Branching stop op
    auto reduce_const = opset5::Constant::create(element::i64, Shape{3}, {1, 2, 3});
    auto reduce_mean = std::make_shared<opset5::ReduceMean>(conv, reduce_const, true);

    auto conv_1_shape = Shape{weightsShape[0], 1, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(reduce_mean, conv_1_weights, Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    // Multiply will try to propagate a non zero masks of the conv_1 up
    // and the mask should be invalidated by reduce_mean stop op mask
    auto mul = std::make_shared<opset5::Multiply>(conv_1, conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset5::Convolution>(mul, weights_end_conv, Strides(2, 1),
                                                          CoordinateDiff(2, 0),
                                                          CoordinateDiff(2, 0),
                                                          Strides(2, 1));

    auto function = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input}, "RestrictedReduceMeanBranching");

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneStopReduceLayerDown.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST_F(TransformationTestsF, MaskPropagationReshapeUp) {
    auto inputShapes = PartialShape{1, 6, 8, 8};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0),
                                                      CoordinateDiff(2, 0),
                                                      Strides(2, 1));

    auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 6, 64, 1});
    auto reshape = std::make_shared<opset5::Reshape>(conv, reshape_const, true);

    auto conv_1_shape = Shape{6, 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(reshape, conv_1_weights, Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    function = std::make_shared<ngraph::Function>(OutputVector{conv_1}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros({
                                                   weightsShape[0] - 3,
                                                   weightsShape[1],
                                                   weightsShape[2],
                                                   weightsShape[3],
                                                  }, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                          CoordinateDiff(2, 0),
                                                          CoordinateDiff(2, 0),
                                                          Strides(2, 1));

        auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 3, 64, 1});
        auto reshape = std::make_shared<opset5::Reshape>(conv, reshape_const, true);

        auto conv_1_shape = Shape{6, 3, 1, 1};
        auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset5::Convolution>(reshape, conv_1_weights, Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        function_ref = std::make_shared<ngraph::Function>(OutputVector{conv_1}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeUp.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, MaskPropagationReshapeUpWithShapeOf) {
    auto inputShapes = PartialShape{1, 6, 8, 8};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0),
                                                      CoordinateDiff(2, 0),
                                                      Strides(2, 1));

    auto shape_of_conv = std::make_shared<opset5::ShapeOf>(conv);
    auto reshape = std::make_shared<opset5::Reshape>(conv, shape_of_conv, true);

    auto conv_1_shape = Shape{6, 6, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(reshape, conv_1_weights, Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    function = std::make_shared<ngraph::Function>(OutputVector{conv_1}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros({
                                                     weightsShape[0] - 3,
                                                     weightsShape[1],
                                                     weightsShape[2],
                                                     weightsShape[3],
                                                    }, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                          CoordinateDiff(2, 0),
                                                          CoordinateDiff(2, 0),
                                                          Strides(2, 1));

        auto shape_of_conv = std::make_shared<opset5::ShapeOf>(conv);
        auto reshape = std::make_shared<opset5::Reshape>(conv, shape_of_conv, true);

        auto conv_1_shape = Shape{6, 6, 1, 1};
        auto conv_1_weights = create_constant_with_zeros({
                                                     conv_1_shape[0],
                                                     conv_1_shape[1] - 3,
                                                     conv_1_shape[2],
                                                     conv_1_shape[3],
                                                    }, {{}, {}, {}, {}});
        auto conv_1 = std::make_shared<opset5::Convolution>(reshape, conv_1_weights, Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));

        function_ref = std::make_shared<ngraph::Function>(OutputVector{conv_1}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeUpWithShapeOf.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, MaskPropagationReshapeDown) {
    auto inputShapes = PartialShape{1, 1, 24, 24};
    auto weightsShape = Shape{8, 1, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{}, {}, {}, {}});
    auto first_conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));



    auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 8, 576, 1});
    auto reshape = std::make_shared<opset5::Reshape>(first_conv, reshape_const, true);

    auto reshape_conv_weights = create_constant_with_zeros({8, 8, 1, 1}, {{1, 2, 3}, {}, {}, {}});
    auto reshape_conv = std::make_shared<opset5::Convolution>(reshape, reshape_conv_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

    auto reshape_const_1 = opset5::Constant::create(element::i64, Shape{4}, {1, 8, 24, 24});
    auto reshape_1 = std::make_shared<opset5::Reshape>(reshape_conv, reshape_const_1, true);

    auto mul = std::make_shared<opset5::Multiply>(first_conv, reshape_1);

    auto last_conv_weights = create_constant_with_zeros({8, 8, 8, 8}, {{1, 2, 3}, {}, {}, {}});
    auto last_conv = std::make_shared<opset5::Convolution>(mul, last_conv_weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    function = std::make_shared<ngraph::Function>(OutputVector{last_conv}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros({
                                                    weightsShape[0] - 3,
                                                    weightsShape[1],
                                                    weightsShape[2],
                                                    weightsShape[3],
                                                   }, {{}, {}, {}, {}});
        auto first_conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                CoordinateDiff(2, 0),
                                                                CoordinateDiff(2, 0),
                                                                Strides(2, 1));

        auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 5, 576, 1});
        auto reshape = std::make_shared<opset5::Reshape>(first_conv, reshape_const, true);

        auto reshape_conv_weights = create_constant_with_zeros({5, 5, 1, 1}, {{}, {}, {}, {}});
        auto reshape_conv = std::make_shared<opset5::Convolution>(reshape, reshape_conv_weights,
                                                                 Strides(2, 1),
                                                                 CoordinateDiff(2, 0),
                                                                 CoordinateDiff(2, 0),
                                                                 Strides(2, 1));

        auto reshape_const_1 = opset5::Constant::create(element::i64, Shape{4}, {1, 5, 24, 24});
        auto reshape_1 = std::make_shared<opset5::Reshape>(reshape_conv, reshape_const_1, true);

        auto mul = std::make_shared<opset5::Multiply>(first_conv, reshape_1);

        auto last_conv_weights = create_constant_with_zeros({8, 5, 8, 8}, {{1, 2, 3}, {}, {}, {}});
        auto last_conv = std::make_shared<opset5::Convolution>(mul, last_conv_weights,
                                                               Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));

        function_ref = std::make_shared<ngraph::Function>(OutputVector{last_conv}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationReshapeDown.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(reshape_conv_weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(reshape_conv->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(last_conv_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)),  Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST(TransformationTests, MaskPropagationStopReshapeUp) {
    auto inputShapes = PartialShape{1, 6, 8, 8};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0),
                                                      CoordinateDiff(2, 0),
                                                      Strides(2, 1));

    auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 3, 128, 1});
    auto reshape = std::make_shared<opset5::Reshape>(conv, reshape_const, true);

    auto conv_1_shape = Shape{6, 3, 1, 1};
    auto conv_1_weights = create_constant_with_zeros(conv_1_shape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(reshape, conv_1_weights, Strides(2, 1),
                                                        CoordinateDiff(2, 0),
                                                        CoordinateDiff(2, 0),
                                                        Strides(2, 1));

    auto function = std::make_shared<ngraph::Function>(OutputVector{conv_1}, ParameterVector{input});
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationStopReshapeUp.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv_1->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST(TransformationTests, MaskPropagationStopReshapeDown) {
    auto inputShapes = PartialShape{1, 1, 24, 24};
    auto weightsShape = Shape{8, 1, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto first_conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));



    auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 32, 12, 12});
    auto reshape = std::make_shared<opset5::Reshape>(first_conv, reshape_const, true);

    auto reshape_conv_weights = create_constant_with_zeros({8, 32, 13, 13}, {{1, 2, 3}, {}, {}, {}});
    auto reshape_conv = std::make_shared<opset5::Convolution>(reshape, reshape_conv_weights,
                                                             Strides(2, 1),
                                                             CoordinateDiff(2, 12),
                                                             CoordinateDiff(2, 12),
                                                             Strides(2, 1));


    auto mul = std::make_shared<opset5::Multiply>(first_conv, reshape_conv);

    auto last_conv_weights = create_constant_with_zeros({8, 8, 8, 8}, {{1, 2, 3}, {}, {}, {}});
    auto last_conv = std::make_shared<opset5::Convolution>(mul, last_conv_weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    auto function = std::make_shared<ngraph::Function>(OutputVector{last_conv}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationStopReshapeDown.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(reshape_conv_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(reshape_conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(last_conv_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST(TransformationTests, MaskPropagationWrongDimsElementwise) {
    auto inputShapes = PartialShape{1, 1, 24, 24};
    auto weightsShape = Shape{8, 1, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto first_conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));


    auto branch_conv_weights = create_constant_with_zeros({32, 8, 2, 2}, {{1, 2, 3}, {}, {}, {}});
    auto branch_conv = std::make_shared<opset5::Convolution>(first_conv, branch_conv_weights,
                                                             Strides(2, 2),
                                                             CoordinateDiff(2, 0),
                                                             CoordinateDiff(2, 0),
                                                             Strides(2, 1));

    auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 32, 12, 12});
    auto reshape = std::make_shared<opset5::Reshape>(first_conv, reshape_const, true);

    auto mul = std::make_shared<opset5::Multiply>(branch_conv, reshape);

    auto last_conv_weights = create_constant_with_zeros({8, 32, 8, 8}, {{1, 2, 3}, {}, {}, {}});
    auto last_conv = std::make_shared<opset5::Convolution>(mul, last_conv_weights,
                                                           Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    auto function = std::make_shared<ngraph::Function>(OutputVector{last_conv}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "MaskPropagationWrongDimsElementwise.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(branch_conv_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(branch_conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(last_conv_weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)),  Mask({{}, {}, {}, {}}));
}


TEST_F(TransformationTestsF, PruneSEBlock) {
    auto inputShapes = PartialShape{1, 6, 16, 16};
    auto weightsShape = Shape{6, 6, 1, 1};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto first_conv_weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto first_conv = std::make_shared<opset5::Convolution>(input, first_conv_weights, Strides(2, 1),
                                                            CoordinateDiff(2, 0),
                                                            CoordinateDiff(2, 0),
                                                            Strides(2, 1));
    auto reduce_const = opset5::Constant::create(element::i64, Shape{2}, {2, 3});
    auto reduce_mean = std::make_shared<opset5::ReduceMean>(first_conv, reduce_const, false);


    auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 6, 1, 1});
    auto reshape = std::make_shared<opset5::Reshape>(reduce_mean, reshape_const, true);

    auto se_conv_0_shape = Shape{weightsShape[0], weightsShape[0], 1, 1};
    auto se_conv_0_weights = create_constant_with_zeros(se_conv_0_shape, {{1, 2, 3}, {}, {}, {}});
    auto se_conv_0 = std::make_shared<opset5::Convolution>(reshape, se_conv_0_weights, Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));
    auto se_conv_1_shape = Shape{weightsShape[0], weightsShape[0], 1, 1};
    auto se_conv_1_weights = create_constant_with_zeros(se_conv_0_shape, {{1, 2, 3}, {}, {}, {}});
    auto se_conv_1 = std::make_shared<opset5::Convolution>(se_conv_0, se_conv_1_weights, Strides(2, 1),
                                                           CoordinateDiff(2, 0),
                                                           CoordinateDiff(2, 0),
                                                           Strides(2, 1));

    auto mul = std::make_shared<opset5::Multiply>(se_conv_1, first_conv);

    auto end_conv_shape = Shape{weightsShape[1], weightsShape[0], 1, 1};
    auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{1, 2, 3}, {}, {}, {}});
    auto end_conv = std::make_shared<opset5::Convolution>(mul, weights_end_conv, Strides(2, 1),
                                                                                 CoordinateDiff(2, 0),
                                                                                 CoordinateDiff(2, 0),
                                                                                 Strides(2, 1));

    function = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
        auto first_conv_weights = create_constant_with_zeros({
                                                              weightsShape[0] - 3,
                                                              weightsShape[1],
                                                              weightsShape[2],
                                                              weightsShape[3]
                                                             }, {{}, {}, {}, {}});
        auto first_conv = std::make_shared<opset5::Convolution>(input, first_conv_weights, Strides(2, 1),
                                                                CoordinateDiff(2, 0),
                                                                CoordinateDiff(2, 0),
                                                                Strides(2, 1));
        auto reduce_const = opset5::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce_mean = std::make_shared<opset5::ReduceMean>(first_conv, reduce_const, false);

        auto reshape_const = opset5::Constant::create(element::i64, Shape{4}, {1, 3, 1, 1});
        auto reshape = std::make_shared<opset5::Reshape>(reduce_mean, reshape_const, true);

        auto se_conv_0_shape = Shape{weightsShape[0] - 3, weightsShape[0] -3, 1, 1};
        auto se_conv_0_weights = create_constant_with_zeros(se_conv_0_shape, {{}, {}, {}, {}});
        auto se_conv_0 = std::make_shared<opset5::Convolution>(reshape, se_conv_0_weights, Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));
        auto se_conv_1_shape = Shape{weightsShape[0] - 3, weightsShape[0] - 3, 1, 1};
        auto se_conv_1_weights = create_constant_with_zeros(se_conv_0_shape, {{}, {}, {}, {}});
        auto se_conv_1 = std::make_shared<opset5::Convolution>(se_conv_0, se_conv_1_weights, Strides(2, 1),
                                                               CoordinateDiff(2, 0),
                                                               CoordinateDiff(2, 0),
                                                               Strides(2, 1));

        auto mul = std::make_shared<opset5::Multiply>(se_conv_1, first_conv);

        auto end_conv_shape = Shape{weightsShape[1], weightsShape[0] - 3, 1, 1};
        auto weights_end_conv = create_constant_with_zeros(end_conv_shape, {{}, {}, {}, {}});
        auto end_conv = std::make_shared<opset5::Convolution>(mul, weights_end_conv, Strides(2, 1),
                                                                                     CoordinateDiff(2, 0),
                                                                                     CoordinateDiff(2, 0),
                                                                                     Strides(2, 1));

        function_ref = std::make_shared<ngraph::Function>(OutputVector{end_conv}, ParameterVector{input}, "SEBlock");
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneSEBlock.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(first_conv_weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {}, {}, {}}));
    compare_masks(*getMask(first_conv->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(se_conv_0_weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(se_conv_0->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(se_conv_1_weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2, 3}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(se_conv_1->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {1, 2, 3}, {}, {}}));
    compare_masks(*getMask(end_conv->output(0)),  Mask({{}, {}, {}, {}}));
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, PropagateMasksLinear) {
    const auto linear_input_features = 62 * 62 * 6;
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_linear_shape{linear_input_features, 100};
    Shape weights_last_linear_shape{100, 10};

    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{0, 1, 2}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto reshape_const = opset5::Constant::create(element::i64, Shape{2}, {1, linear_input_features});
    auto reshape = std::make_shared<opset5::Reshape>(relu, reshape_const, true);

    auto weights_linear = create_constant_with_zeros(weights_linear_shape, {{}, {0, 1, 2}});
    auto linear = std::make_shared<opset5::MatMul>(reshape, weights_linear);

    // Do net search 0 dim zeros by now
    // Check stop mask prop for outer dim (1)
    auto weights_last_linear = create_constant_with_zeros(weights_last_linear_shape, {{3, 4, 5}, {2, 3, 4}});
    auto last_linear = std::make_shared<opset5::MatMul>(linear, weights_last_linear);
    function = std::make_shared<Function>(NodeVector{last_linear}, ParameterVector{input});

    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights = create_constant_with_zeros({
                                                   weights_shape[0] - 3,
                                                   weights_shape[1],
                                                   weights_shape[2],
                                                   weights_shape[3],
                                                   }, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        auto relu = std::make_shared<opset5::Relu>(conv);

        auto reshape_const = opset5::Constant::create(element::i64, Shape{2}, {1, linear_input_features / 2});
        auto reshape = std::make_shared<opset5::Reshape>(relu, reshape_const, true);

        auto weights_linear = create_constant_with_zeros({
                                                   weights_linear_shape[0] / 2,
                                                   weights_linear_shape[1] - 3,
                                                   }, {{}, {}});
        auto linear = std::make_shared<opset5::MatMul>(reshape, weights_linear);

        auto weights_last_linear = create_constant_with_zeros({
                                                   weights_last_linear_shape[0] - 3,
                                                   weights_last_linear_shape[1],
                                                   }, {{}, {}});
        auto last_linear = std::make_shared<opset5::MatMul>(linear, weights_last_linear);
        function_ref = std::make_shared<Function>(NodeVector{last_linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateMasksLinear.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{0, 1, 2}, {}, {}, {}}));
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
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST(TransformationTests, PruneLinearUp) {
    const auto linear_input_features = 6 * 2 * 2;
    auto inputShapes = PartialShape{1, 6, 2, 2};
    auto weightsShape = Shape{6, 6, 1, 1};
    auto linearShape = Shape{linear_input_features, linear_input_features};
    auto lastLinearShape = Shape{10, linear_input_features};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                      CoordinateDiff(2, 0),
                                                                      CoordinateDiff(2, 0),
                                                                      Strides(2, 1));

    auto reshape_const = opset5::Constant::create(element::i64, Shape{2}, {1, linear_input_features});
    auto reshape = std::make_shared<opset5::Reshape>(conv, reshape_const, true);

    auto linear_mask = Mask();
    auto outer_dim_zeros = std::set<uint64_t>();
    for (auto i = 0; i < linear_input_features / 2; ++i)
        outer_dim_zeros.insert(i);
    linear_mask.push_back({10, 11});
    linear_mask.push_back(outer_dim_zeros);
    auto linear_const = create_constant_with_zeros(linearShape, linear_mask);
    auto linear = std::make_shared<opset5::MatMul>(reshape, linear_const);

    auto add_mask = Mask();
    add_mask.push_back({});
    add_mask.push_back(outer_dim_zeros);
    auto add_const = create_constant_with_zeros({1, linear_input_features}, add_mask);
    auto add = std::make_shared<opset5::Add>(linear, add_const);
    auto add_const_1 = create_constant_with_zeros({1, linear_input_features}, add_mask);
    auto add_1 = std::make_shared<opset5::Add>(add, add_const_1);
    auto add_2 = std::make_shared<opset5::Add>(add_1, reshape);

    auto bad_add_const = create_constant_with_zeros({1, linear_input_features}, {{}, {}});
    auto bad_add = std::make_shared<opset5::Add>(add_2, bad_add_const);

    auto weights_end_linear = create_constant_with_zeros(lastLinearShape, {{1, 2, 3}, {3, 4, 6}});
    auto last_linear = std::make_shared<opset5::MatMul>(bad_add, weights_end_linear, false, true);
    auto function = std::make_shared<ngraph::Function>(OutputVector{last_linear}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneLinearUp.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_linear.get_node_shared_ptr()->output(0)),  Mask({{}, {}}));
    compare_masks(*getMask(last_linear->output(0)),  Mask({{}, {}}));
}


TEST(TransformationTests, PruneConvUpShort) {
    const auto linear_input_features = 6 * 2 * 2;
    auto inputShapes = PartialShape{1, 6, 2, 2};
    auto convShape = Shape{1, 6, 2, 2};
    auto weightsShape = Shape{6, 6, 1, 1};
    auto lastLinearShape = Shape{10, linear_input_features};

    auto input = std::make_shared<opset5::Parameter>(element::f32, inputShapes);
    auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                                      CoordinateDiff(2, 0),
                                                                      CoordinateDiff(2, 0),
                                                                      Strides(2, 1));


    auto conv_1_const = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
    auto conv_1 = std::make_shared<opset5::Convolution>(conv, conv_1_const, Strides(2, 1),
                                                                        CoordinateDiff(2, 0),
                                                                        CoordinateDiff(2, 0),
                                                                        Strides(2, 1));

    auto add_const = create_constant_with_zeros(convShape, {{}, {1, 2, 3}, {}, {}});
    auto add = std::make_shared<opset5::Add>(conv_1, add_const);
    auto add_const_1 = create_constant_with_zeros(convShape, {{}, {1, 2, 3}, {}, {}});
    auto add_1 = std::make_shared<opset5::Add>(add, add_const_1);
    auto add_2 = std::make_shared<opset5::Add>(add_1, conv);

    auto bad_add_const = create_constant_with_zeros(convShape, {{}, {}, {}, {}});
    auto bad_add = std::make_shared<opset5::Add>(add_2, bad_add_const);

    auto weights_end_conv = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {1, 2, 3}, {}, {}});
    auto last_conv = std::make_shared<opset5::Convolution>(bad_add, weights_end_conv, Strides(2, 1),
                                                                        CoordinateDiff(2, 0),
                                                                        CoordinateDiff(2, 0),
                                                                        Strides(2, 1));

    auto function = std::make_shared<ngraph::Function>(OutputVector{last_conv}, ParameterVector{input});

    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneConvUpShort.svg").run_on_function(function);

    pass::Manager m;
    m.register_pass<pass::Pruning>();
    m.run_passes(function);

    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)),  Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(weights_end_conv.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(last_conv->output(0)),  Mask({{}, {}, {}, {}}));
}

TEST_F(TransformationTestsF, PruneMasksMatMulColsStopRowsUp) {
    const auto linear_input_features = 62 * 62;
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_linear_shape{linear_input_features, 100};
    Shape weights_last_linear_shape{100, 10};

    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{0, 1, 2}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto reshape_const = opset5::Constant::create(element::i64, Shape{3}, {1, 6, linear_input_features});
    auto reshape = std::make_shared<opset5::Reshape>(relu, reshape_const, true);

    auto weights_linear = create_constant_with_zeros(weights_linear_shape, {{}, {0, 1, 2}});
    auto linear = std::make_shared<opset5::MatMul>(reshape, weights_linear);

    // Do net search 0 dim zeros by now
    auto weights_last_linear = create_constant_with_zeros(weights_last_linear_shape, {{3, 4, 5}, {}});
    auto last_linear = std::make_shared<opset5::MatMul>(linear, weights_last_linear);
    function = std::make_shared<Function>(NodeVector{last_linear}, ParameterVector{input});

    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights = create_constant_with_zeros(weights_shape, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        auto relu = std::make_shared<opset5::Relu>(conv);

        auto reshape_const = opset5::Constant::create(element::i64, Shape{3}, {1, 6, linear_input_features});
        auto reshape = std::make_shared<opset5::Reshape>(relu, reshape_const, true);

        auto weights_linear = create_constant_with_zeros({
                                                   weights_linear_shape[0],
                                                   weights_linear_shape[1] - 3,
                                                   }, {{}, {}});
        auto linear = std::make_shared<opset5::MatMul>(reshape, weights_linear);

        auto weights_last_linear = create_constant_with_zeros({
                                                   weights_last_linear_shape[0] - 3,
                                                   weights_last_linear_shape[1],
                                                   }, {{}, {}});
        auto last_linear = std::make_shared<opset5::MatMul>(linear, weights_last_linear);
        function_ref = std::make_shared<Function>(NodeVector{last_linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneMasksMatMulColsStopRowsUp.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {}, {}, {}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask{{}, {}, {}});
    compare_masks(*getMask(reshape->output(0)), Mask{{}, {}, {}});
    compare_masks(*getMask(weights_linear.get_node_shared_ptr()->output(0)), Mask({{}, {0, 1, 2}}));
    compare_masks(*getMask(linear->output(0)), Mask{{}, {}, {0, 1, 2}});
    compare_masks(*getMask(weights_last_linear.get_node_shared_ptr()->output(0)), Mask{{0, 1, 2}, {}});
    compare_masks(*getMask(last_linear->output(0)), Mask{{}, {}, {}});
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, PruneMasksMatMulRowsStopColsUp) {
    // Checks rows matmul pruning + transpose input in matmul
    const auto linear_input_features = 62 * 62;
    Shape input_shape{1, 3, 64, 64};
    Shape weights_shape{6, 3, 3, 3};
    Shape weights_linear_shape{linear_input_features, 100};
    Shape weights_last_linear_shape{10, 6};

    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{0, 1, 2}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto reshape_const = opset5::Constant::create(element::i64, Shape{3}, {1, 6, linear_input_features});
    auto reshape = std::make_shared<opset5::Reshape>(relu, reshape_const, true);

    auto weights_linear = create_constant_with_zeros(weights_linear_shape, {{3, 4, 5}, {3, 4}});
    auto linear = std::make_shared<opset5::MatMul>(reshape, weights_linear);

    // Do net search this zeros by now
    auto weights_last_linear = create_constant_with_zeros(weights_last_linear_shape, {{}, {3, 4, 5}});
    // To prune rows we should transpose featuremap. Did it by transpose_a = true MatMul constructor attr
    auto last_linear = std::make_shared<opset5::MatMul>(linear, weights_last_linear, true, true);
    function = std::make_shared<Function>(NodeVector{last_linear}, ParameterVector{input});

    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights = create_constant_with_zeros({
                                                  weights_shape[0] - 3,
                                                  weights_shape[1],
                                                  weights_shape[2],
                                                  weights_shape[3],
                                                  }, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        auto relu = std::make_shared<opset5::Relu>(conv);

        auto reshape_const = opset5::Constant::create(element::i64, Shape{3}, {1, 3, linear_input_features});
        auto reshape = std::make_shared<opset5::Reshape>(relu, reshape_const, true);

        auto weights_linear = create_constant_with_zeros({
                                                   weights_linear_shape[0],
                                                   weights_linear_shape[1],
                                                   }, {{}, {}});
        auto linear = std::make_shared<opset5::MatMul>(reshape, weights_linear);

        auto weights_last_linear = create_constant_with_zeros({weights_last_linear_shape[0],
                                                               weights_last_linear_shape[1] - 3}, {{}, {}});
        // To prune rows we should transpose featuremap. Did it by transpose_a = true MatMul constructor attr
        auto last_linear = std::make_shared<opset5::MatMul>(linear, weights_last_linear, true, true);
        function_ref = std::make_shared<Function>(NodeVector{last_linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PruneMasksMatMulRowsStopColsUp.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{0, 1, 2}, {}, {}, {}}));
    compare_masks(*getMask(conv->output(0)), Mask({{}, {0, 1, 2}, {}, {}}));
    compare_masks(*getMask(relu->output(0)), Mask({{}, {0, 1, 2}, {}, {}}));

    compare_masks(*getMask(reshape_const->output(0)), Mask{{}, {0, 1, 2}, {}});
    compare_masks(*getMask(reshape->output(0)), Mask{{}, {0, 1, 2}, {}});
    compare_masks(*getMask(weights_linear.get_node_shared_ptr()->output(0)), Mask{{}, {}});
    compare_masks(*getMask(linear->output(0)), Mask{{}, {0, 1, 2}, {}});
    compare_masks(*getMask(weights_last_linear.get_node_shared_ptr()->output(0)), Mask{{}, {0, 1, 2}});
    compare_masks(*getMask(last_linear->output(0)), Mask{{}, {}, {}});
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}


TEST_F(TransformationTestsF, PropagateFlattenUp) {
    // Propagate Flatten down is the same as in
    // PruneLinearIsClosingAndInGroup test
    using nested_vector = std::vector<std::set<uint64_t>>;
    constexpr auto linear_input_features = 6 * 8 * 8;
    Shape input_shape{1, 3, 8, 8};
    Shape weights_shape{6, 3, 1, 1};
    Shape weights_linear_shape{linear_input_features, 100};

    auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
    auto weights = create_constant_with_zeros(weights_shape, {{0, 1, 2}, {}, {}, {}});
    auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                      CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
    auto relu = std::make_shared<opset5::Relu>(conv);

    auto reshape_const = opset5::Constant::create(element::i64, Shape{2}, {1, linear_input_features});
    auto reshape = std::make_shared<opset5::Reshape>(relu, reshape_const, true);

    // Skip just one zero in dim should lead to
    // whole dimension invalidating.
    auto add_zeros = std::set<uint64_t>();
    for (size_t i = 1; i < linear_input_features / 2; i++)
        add_zeros.insert(i);
    auto add_mask = nested_vector();
    add_mask.push_back({});
    add_mask.push_back(add_zeros);
    auto weights_add = create_constant_with_zeros({1, linear_input_features}, Mask(add_mask));
    auto add = std::make_shared<opset5::Add>(reshape, weights_add);

    auto weights_linear = create_constant_with_zeros(weights_linear_shape, {{}, {0, 1, 2}});
    auto linear = std::make_shared<opset5::MatMul>(add, weights_linear);

    function = std::make_shared<Function>(NodeVector{linear}, ParameterVector{input});
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, input_shape);
        auto weights = create_constant_with_zeros({
                                                   weights_shape[0] - 2,
                                                   weights_shape[1],
                                                   weights_shape[2],
                                                   weights_shape[3],
                                                   }, {{}, {}, {}, {}});
        auto conv = std::make_shared<opset5::Convolution>(input, weights, Strides(2, 1),
                                                          CoordinateDiff(2, 0), CoordinateDiff(2, 0), Strides(2, 1));
        auto relu = std::make_shared<opset5::Relu>(conv);

        auto reshape_const = opset5::Constant::create(element::i64, Shape{2}, {1, 2 * linear_input_features / 3});
        auto reshape = std::make_shared<opset5::Reshape>(relu, reshape_const, true);

        auto weights_add = create_constant_with_zeros({1, 2 * linear_input_features / 3}, Mask{{}, {}});
        auto add = std::make_shared<opset5::Add>(reshape, weights_add);

        auto weights_linear = create_constant_with_zeros({
                                                   2 * weights_linear_shape[0] / 3,
                                                   weights_linear_shape[1],
                                                   }, {{}, {}});
        auto linear = std::make_shared<opset5::MatMul>(add, weights_linear);

        function_ref = std::make_shared<Function>(NodeVector{linear}, ParameterVector{input});
    }
    if (VISUALIZE_TESTS_TREE)
        ngraph::pass::VisualizeTree(std::string(VISUALIZE_TREE_ROOT) + "PropagateFlattenUp.svg").run_on_function(function);
    {
        pass::Manager m;
        m.register_pass<pass::InitMasks>();
        m.register_pass<pass::PropagateMasks>();
        m.run_passes(function);
    }
    compare_masks(*getMask(weights.get_node_shared_ptr()->output(0)),  Mask({{1, 2}, {}, {}, {}}));
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
    {
        pass::Manager m;
        m.register_pass<pass::ShrinkWeights>();
        m.run_passes(function);
    }
    disable_rt_info_check();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
