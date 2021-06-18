// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "insert_transpose_before_matmul.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {

using ConstPtr = std::shared_ptr<ngraph::opset7::Constant>;
using MatMulPtr = std::shared_ptr<ngraph::opset7::MatMul>;
using ParameterPtr = std::shared_ptr<ngraph::opset7::Parameter>;
using ReshapePtr = std::shared_ptr<ngraph::opset7::Reshape>;
using ResultPtr = std::shared_ptr<ngraph::opset7::Result>;
using TransposePtr = std::shared_ptr<ngraph::opset7::Transpose>;


TEST(TransformationTests, InsertTransposeBeforeMatmulTestShapeNotSupported) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{2, 9};

    {
        ParameterPtr input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        ConstPtr new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {9, 2});
        ReshapePtr reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

        ConstPtr constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 1});
        MatMulPtr matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_operation, constant);

        ResultPtr result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeBeforeMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        ParameterPtr input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        ConstPtr new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {9, 2});
        ReshapePtr reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

        ConstPtr constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 1});
        MatMulPtr matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_operation, constant);

        ResultPtr result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeBeforeMatmulTestReshapeInOutEq) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{9, 2};

    {
        ParameterPtr input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        ConstPtr new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {9, 2});
        ReshapePtr reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

        ConstPtr constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 1});
        MatMulPtr matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_operation, constant);

        ResultPtr result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeBeforeMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        ParameterPtr input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        ConstPtr new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {9, 2});
        ReshapePtr reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

        ConstPtr constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 1});
        MatMulPtr matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_operation, constant);

        ResultPtr result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeBeforeMatmulTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{2, 8};

    {
        ParameterPtr input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        ConstPtr new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {8, 2});
        ReshapePtr reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

        ConstPtr constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 1});
        MatMulPtr matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_operation, constant);

        ResultPtr result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeBeforeMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        ParameterPtr input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        ConstPtr new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {8, 2});
        ReshapePtr reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

        ConstPtr transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2},
                                                                std::vector<size_t>{1, 0});
        TransposePtr transpose_operation = std::make_shared<ngraph::opset7::Transpose>(reshape_operation, transpose_order);

        ConstPtr new_shape_after_transpose = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {8, 2});
        ReshapePtr reshape_after_transpose = std::make_shared<ngraph::opset7::Reshape>(transpose_operation,
                                                                                 new_shape_after_transpose,
                                                                                 false);

        ConstPtr constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 1});
        MatMulPtr matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_after_transpose, constant);

        ResultPtr result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace testing
