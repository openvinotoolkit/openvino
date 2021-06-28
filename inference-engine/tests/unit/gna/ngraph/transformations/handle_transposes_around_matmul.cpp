// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/handle_transposes_around_matmul.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

std::shared_ptr<ngraph::Function> CreateTransposeMatmulFunction(const ngraph::Shape& input_shape,
    const ngraph::Shape& new_shape, const ngraph::Shape& const_shape) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);

    auto new_shape_const = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{new_shape.size()}, new_shape);
    auto reshape = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape_const, false);

    auto transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0});
    auto transpose = std::make_shared<ngraph::opset7::Transpose>(reshape, transpose_order);
    auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{const_shape.size()}, const_shape);
    auto matmul = std::make_shared<ngraph::opset7::MatMul>(transpose, constant);

    auto result = std::make_shared<ngraph::opset7::Result>(matmul);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

std::shared_ptr<ngraph::Function> CreateMatmulFunction(const ngraph::Shape& input_shape,
    const ngraph::Shape& new_shape, const ngraph::Shape& const_shape) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);

    auto new_shape_const = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{new_shape.size()}, new_shape);
    auto reshape = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape_const, false);

    auto new_shape_after_transpose = ngraph::opset7::Constant::create(ngraph::element::i64,
        ngraph::Shape{input_shape.size()}, {new_shape[1], new_shape[0]});
    auto reshape_after_transpose = std::make_shared<ngraph::opset7::Reshape>(reshape,
                                                                             new_shape_after_transpose,
                                                                             false);

    auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{const_shape.size()}, const_shape);
    auto matmul = std::make_shared<ngraph::opset7::MatMul>(reshape_after_transpose, constant);

    auto result = std::make_shared<ngraph::opset7::Result>(matmul);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

TEST(TransformationTests, RemoveTransposeBeforeMatmulTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{1, 8};

    {
        func = CreateTransposeMatmulFunction(data_shape, {2, 4}, {2, 1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleTransposesAroundMatMul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = CreateMatmulFunction(data_shape, {2, 4}, {2, 1});

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, RemoveTransposeBeforeMatmulTestReshapeInOutEq) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{2, 8};

    {
        func = CreateTransposeMatmulFunction(data_shape, {2, 8}, {8, 1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleTransposesAroundMatMul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = CreateTransposeMatmulFunction(data_shape, {2, 8}, {8, 1});

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}