// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/insert_transpose_before_matmul.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {

std::shared_ptr<ngraph::Function> createFunction(const std::vector<size_t>& input_values,
                                                     const std::vector<size_t>& reshape_values,
                                                     const std::vector<size_t>& matmul_values)
{
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, ngraph::Shape(input_values));

    auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, reshape_values);
    auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

    auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, matmul_values);
    auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_operation, constant);

    auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
}

TEST(TransformationTests, InsertTransposeBeforeMatmulTestShapeNotSupported) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunction({2, 9}, {9, 2}, {2, 1});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeBeforeMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createFunction({2, 9}, {9, 2}, {2, 1});

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeBeforeMatmulTestReshapeInOutEq) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{9, 2};

    {
        func = createFunction({9, 2}, {9, 2}, {2, 1});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeBeforeMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createFunction({9, 2}, {9, 2}, {2, 1});

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeBeforeMatmulTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{2, 8};

    {
        func = createFunction({2, 8}, {8, 2}, {2, 1});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeBeforeMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {8, 2});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

        auto transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2},
                                                                std::vector<size_t>{1, 0});
        auto transpose_operation = std::make_shared<ngraph::opset7::Transpose>(reshape_operation, transpose_order);

        auto new_shape_after_transpose = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {8, 2});
        auto reshape_after_transpose = std::make_shared<ngraph::opset7::Reshape>(transpose_operation,
                                                                                 new_shape_after_transpose,
                                                                                 false);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_after_transpose, constant);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeBeforeMatmulTest1_16) {
     std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{1, 16};

    {
        func = createFunction({1, 16}, {8, 2}, {2, 1});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeBeforeMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {8, 2});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

        auto transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2},
                                                                std::vector<size_t>{1, 0});
        auto transpose_operation = std::make_shared<ngraph::opset7::Transpose>(reshape_operation, transpose_order);

        auto new_shape_after_transpose = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {8, 2});
        auto reshape_after_transpose = std::make_shared<ngraph::opset7::Reshape>(transpose_operation,
                                                                                 new_shape_after_transpose,
                                                                                 false);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_after_transpose, constant);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);

}

} // namespace testing
