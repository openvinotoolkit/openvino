// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/common_optimizations/transpose_sinking_general.hpp>

#include <transformations/init_node_info.hpp>
#include <openvino/frontend/manager.hpp>

#include <openvino/opsets/opset9.hpp>

#include <openvino/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

#include <functional>

#include "gtest/gtest.h"

#include "ngraph/pass/visualize_tree.hpp" // DEBUG

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<ov::Model>;

TEST(TransposeSinkingGeneralTests, UnariesTransposesForward) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 10;

    ModelPtr model, original_model, reference_model;
    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

            auto binary = std::make_shared<ov::opset9::Tanh>(transpose0);

            auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<ov::opset9::Transpose>(binary, ng_order1);
        }

        original_model = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<ov::opset9::Tanh>(in_op);
        }

        reference_model = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    model = original_model->clone();

    //
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::TransposeSinkingGeneralForward>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    //
    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(model, reference_model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransposeSinkingGeneralTests, UnariesTransposesBackward) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 10;

    ModelPtr model, original_model, reference_model;
    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

            auto binary = std::make_shared<ov::opset9::Tanh>(transpose0);

            auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<ov::opset9::Transpose>(binary, ng_order1);
        }

        original_model = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<ov::opset9::Tanh>(in_op);
        }

        reference_model = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    model = original_model->clone();

    //
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::TransposeSinkingGeneralBackward>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    //
    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(model, reference_model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransposeSinkingGeneralTests, UnariesTransposesGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 2;

    ModelPtr model, original_model, reference_model;
    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

            auto binary = std::make_shared<ov::opset9::Tanh>(transpose0);

            auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<ov::opset9::Transpose>(binary, ng_order1);
        }

        original_model = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<ov::opset9::Tanh>(in_op);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        reference_model = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    model = original_model->clone();

    //
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::TransposeSinkingGeneral>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    //
    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(model, reference_model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransposeSinkingGeneralTests, BinaryTransposesGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_binary_ops = 3;

    ModelPtr model, original_model, reference_model;
    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_binary_ops; ++i) {
            auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
            auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

            in_op = std::make_shared<ov::opset9::Add>(in_op, transpose1);
        }

        original_model = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_binary_ops; ++i) {
            auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
            in_op = std::make_shared<ov::opset9::Add>(in_op, in_constant);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        reference_model = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    model = original_model->clone();

    //
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::TransposeSinkingGeneral>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    //
    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(model, reference_model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransposeSinkingGeneralTests, ConcatTransposesGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    const size_t num_concat_ops = 3;
    const size_t num_concat_inputs = 2;

    ModelPtr model, original_model, reference_model;
    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_concat_ops; ++i) {
            ov::OutputVector concat_inputs;
            concat_inputs.push_back(in_op);
            for (size_t j = 1; j < num_concat_inputs; ++j) {
                auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
                auto ng_order1 =
                    std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
                auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);
                concat_inputs.push_back(transpose1);
            }
            in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);
        }

        original_model = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_concat_ops; ++i) {
            ov::OutputVector concat_inputs;

            concat_inputs.push_back(in_op);

            for (size_t j = 1; j < num_concat_inputs; ++j) {
                auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
                concat_inputs.push_back(in_constant);
            }
            in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 2);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        reference_model = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    model = original_model->clone();

    //
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::TransposeSinkingGeneral>();
    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    //
    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(model, reference_model);
    ASSERT_TRUE(result.valid) << result.message;
}
