// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/swap_input_matmul_gna.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {

TEST(TransformationTests, SwapInputMatMulTestValidConstShape) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{8, 8};

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1, 8}, {1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(constant, input_params);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        reference_func = ngraph::clone_function(*func);

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SwapInputMatMul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SwapInputMatMulTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{8, 8};

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{16, 8}, {1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(constant, input_params);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SwapInputMatMul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{16, 8}, {1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(input_params, constant, 1, 1);

        auto transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2},
                                                                std::vector<size_t>{1, 0});
        auto transpose_operation = std::make_shared<ngraph::opset7::Transpose>(matmul_operation, transpose_order);

        auto result = std::make_shared<ngraph::opset7::Result>(transpose_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SwapInputMatMulTestFakeQuantize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{8, 8};

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{16, 8}, {1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(constant, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(fake_quantize_op, input_params);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SwapInputMatMul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{16, 8}, {1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(constant, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(input_params, fake_quantize_op, 1 , 1);

        auto transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2},
                                                                    std::vector<size_t>{1, 0});
        auto transpose_operation = std::make_shared<ngraph::opset7::Transpose>(matmul_operation, transpose_order);

        auto result = std::make_shared<ngraph::opset7::Result>(transpose_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SwapInputMatMulTestRank1) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{8, 8};

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{8}, {1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(constant, input_params);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        reference_func = ngraph::clone_function(*func);

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SwapInputMatMul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace testing
