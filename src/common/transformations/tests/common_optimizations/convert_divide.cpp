// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_divide.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ConvertDivide) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto divide_constant = opset1::Constant::create(element::f32, Shape{1}, {1.5});
        auto divide = std::make_shared<opset1::Divide>(data, divide_constant);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto divide_constant = opset1::Constant::create(element::f32, Shape{1}, {1. / 1.5});
        auto mul = std::make_shared<opset1::Multiply>(data, divide_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideInverse) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto divide_constant = opset1::Constant::create(element::f32, Shape{1}, {1});
        auto divide = std::make_shared<opset1::Divide>(divide_constant, data);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto constant = opset1::Constant::create(element::f32, Shape{}, {-1.0});
        auto pow = std::make_shared<opset1::Power>(data, constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{pow}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideNegative) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::i32, Shape{3, 1, 2});
        auto divide_constant = opset1::Constant::create(element::i32, Shape{1}, {2});
        auto divide = std::make_shared<opset1::Divide>(data, divide_constant);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::i32, Shape{3, 1, 2});
        auto divide_constant = opset1::Constant::create(element::i32, Shape{1}, {2});
        auto divide = std::make_shared<opset1::Divide>(data, divide_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideScalar) {
    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto divide = std::make_shared<opset1::Divide>(data1, data2);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data1, data2});

        OPENVINO_ASSERT(divide->get_output_partial_shape(0).rank().get_length() == 0);

        manager.register_pass<ov::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto pow_input = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto pow = std::make_shared<opset1::Power>(pow_input, opset1::Constant::create(element::f32, Shape{}, {-1}));
        auto mul = std::make_shared<opset1::Multiply>(data, pow);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data, pow_input});

        OPENVINO_ASSERT(mul->get_output_partial_shape(0).rank().get_length() == 0);
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideWithConstantPositive) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto divide_constant = opset1::Constant::create(element::f32, Shape{}, {1.5});
        auto divide = std::make_shared<opset1::Divide>(data, divide_constant);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data});
        manager.register_pass<ov::pass::ConvertDivideWithConstant>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto divide_constant = opset1::Constant::create(element::f32, Shape{}, {1. / 1.5});
        auto mul = std::make_shared<opset1::Multiply>(data, divide_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideWithConstantNegative) {
    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto divide = std::make_shared<opset1::Divide>(data1, data2);

        model = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data1, data2});
        manager.register_pass<ov::pass::ConvertDivideWithConstant>();
    }

    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto divide = std::make_shared<opset1::Divide>(data1, data2);

        model_ref = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data1, data2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideFP16ShapeOfSubgraphNegative) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f16, Shape{1, 3, 22, 22});
        auto gather = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, {2, 3});
        auto convert = std::make_shared<opset1::Convert>(gather, element::f16);
        auto divide_constant = opset1::Constant::create(element::f16, Shape{1}, {0.5});
        auto divide = std::make_shared<opset1::Divide>(convert, divide_constant);
        auto convert_after = std::make_shared<opset1::Convert>(divide, element::i32);

        opset1::Interpolate::Attributes interp_attr;
        interp_attr.antialias = false;
        interp_attr.axes = {2, 3};
        interp_attr.mode = "nearest";
        interp_attr.pads_begin = {0, 0, 0, 0};
        interp_attr.pads_end = {0, 0, 0, 0};

        auto interpolate = std::make_shared<opset1::Interpolate>(data, convert_after, interp_attr);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{data});

        manager.register_pass<ov::pass::MarkDividesInShapeSubgraphs>();
        manager.register_pass<ov::pass::ConvertDivide>();
    }
}

TEST_F(TransformationTestsF, ConvertDivide_If) {
    {
        auto data1 = opset1::Constant::create(element::f16, Shape{1}, {5});
        auto data2 = opset1::Constant::create(element::f16, Shape{1}, {1});
        auto data = std::make_shared<opset1::Parameter>(element::f16, Shape{1, 3, 22, 22});
        auto divide = std::make_shared<opset8::Divide>(data, data2);
        auto convert_after = std::make_shared<opset1::Convert>(divide, element::i32);

        opset1::Interpolate::Attributes interp_attr;
        interp_attr.antialias = false;
        interp_attr.axes = {2, 3};
        interp_attr.mode = "nearest";
        interp_attr.pads_begin = {0, 0, 0, 0};
        interp_attr.pads_end = {0, 0, 0, 0};

        auto interpolate = std::make_shared<opset1::Interpolate>(data, convert_after, interp_attr);
        auto then_op_result = std::make_shared<opset1::Result>(interpolate);

        auto body_then_function = std::make_shared<ov::Model>(NodeVector{then_op_result}, ParameterVector{data});

        // create else body
        auto input_else = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ov::Shape{1, 3, 22, 22});
        auto else_op_result = std::make_shared<opset1::Result>(input_else);
        auto body_else_function =
            std::make_shared<ov::Model>(ov::NodeVector{else_op_result}, ov::ParameterVector{input_else});

        // create main graph
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ov::Shape{1, 3, 22, 22});
        auto cond = std::make_shared<opset1::Constant>(element::boolean, Shape{1}, true);
        auto if_op = std::make_shared<ov::opset8::If>(cond);
        if_op->set_then_body(body_then_function);
        if_op->set_else_body(body_else_function);
        if_op->set_input(input, data, input_else);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<opset1::Result>(if_op);

        model = std::make_shared<ov::Model>(NodeVector{if_result}, ParameterVector{input});

        manager.register_pass<ov::pass::MarkDividesInShapeSubgraphs>();
        auto decomp = manager.register_pass<pass::GraphRewrite>();
        decomp->add_matcher<ov::pass::ConvertDivide>();
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertDivideFP16ShapeOfSubgraphNegative2) {
    {
        // This test case checks that MarkDividesInShapeSubgraphs works correctly when Divide is included
        // into precision sensitive and non precision sensitive sub-graphs. So the potential problem here is
        // that MarkDividesInShapeSubgraphs could traverse graph first form "add" output so all nodes including
        // Divide will be marked as visited, but Divide and other nodes must also be visited again because of
        // precision sensitive Interpolate second input. And to handle this MarkDividesInShapeSubgraphs has
        // special visited set for precision sensitive nodes which needs to be tested as well. So in the worst case
        // we will traverse each node twice.
        auto data = std::make_shared<opset1::Parameter>(element::f16, Shape{1, 3, 22, 22});
        auto gather = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, {2, 3});
        auto convert = std::make_shared<opset1::Convert>(gather, element::f16);
        auto divide_constant = opset1::Constant::create(element::f16, Shape{1}, {0.5});
        auto divide = std::make_shared<opset1::Divide>(convert, divide_constant);
        auto convert_after = std::make_shared<opset1::Convert>(divide, element::i32);

        auto data2 = std::make_shared<opset1::Parameter>(element::i32, Shape{2});
        auto add = std::make_shared<opset1::Add>(data2, convert_after);

        opset1::Interpolate::Attributes interp_attr;
        interp_attr.antialias = false;
        interp_attr.axes = {2, 3};
        interp_attr.mode = "nearest";
        interp_attr.pads_begin = {0, 0, 0, 0};
        interp_attr.pads_end = {0, 0, 0, 0};

        auto interpolate = std::make_shared<opset1::Interpolate>(data, convert_after, interp_attr);

        // "add" node specially set as a first output, so MarkDividesInShapeSubgraphs will start graph traversal from
        // it and after all nodes above are visited it will start traverse from "interpolate"
        model = std::make_shared<ov::Model>(NodeVector{add, interpolate}, ParameterVector{data, data2});

        manager.register_pass<ov::pass::MarkDividesInShapeSubgraphs>();
        manager.register_pass<ov::pass::ConvertDivide>();
    }
}

TEST_F(TransformationTestsF, ConvertDivideFP32ShapeOfSubgraphNegative) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 22, 22});
        auto gather = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, {2, 3});
        auto convert = std::make_shared<opset1::Convert>(gather, element::f32);
        auto divide_constant = opset1::Constant::create(element::f32, Shape{1}, {0.5});
        auto divide = std::make_shared<opset1::Divide>(convert, divide_constant);
        auto convert_after = std::make_shared<opset1::Convert>(divide, element::i32);

        opset1::Interpolate::Attributes interp_attr;
        interp_attr.antialias = false;
        interp_attr.axes = {2, 3};
        interp_attr.mode = "nearest";
        interp_attr.pads_begin = {0, 0, 0, 0};
        interp_attr.pads_end = {0, 0, 0, 0};

        auto interpolate = std::make_shared<opset1::Interpolate>(data, convert_after, interp_attr);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{data});

        manager.register_pass<ov::pass::MarkDividesInShapeSubgraphs>();
        manager.register_pass<ov::pass::ConvertDivide>();
    }
}
