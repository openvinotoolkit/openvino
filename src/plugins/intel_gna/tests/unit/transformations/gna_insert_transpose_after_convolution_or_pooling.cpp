// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/insert_transpose_after_convolution_or_pooling.hpp"

namespace testing {

TEST(TransformationTests, InsertTransposeAfterConvOrPoolTestStartConvolution) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_convolution =
            std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, ngraph::Shape{1, 3, 1, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3, 3, 1, 2}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                   weights,
                                                                                   ngraph::Strides{1, 1},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::CoordinateDiff{0, 1},
                                                                                   ngraph::Strides{1, 1});

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 3 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(convolution_operation, new_shape, true);

        auto weights_next_convolution =
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1, 1, 1, 3 * 63}, {1});
        auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                                        weights_next_convolution,
                                                                                        ngraph::Strides{1, 1},
                                                                                        ngraph::CoordinateDiff{0, 0},
                                                                                        ngraph::CoordinateDiff{0, 1},
                                                                                        ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::InsertTransposeAfterConvOrPool>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_convolution =
            std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, ngraph::Shape{1, 3, 1, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3, 3, 1, 2}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                   weights,
                                                                                   ngraph::Strides{1, 1},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::CoordinateDiff{0, 1},
                                                                                   ngraph::Strides{1, 1});

        auto new_shape_out = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 64, 1, 3});
        auto reshape_out_operation =
            std::make_shared<ngraph::opset7::Reshape>(convolution_operation, new_shape_out, false);

        auto transpose = std::make_shared<ngraph::opset7::Transpose>(
            reshape_out_operation,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2}));

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 3 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(transpose, new_shape, true);

        auto weights_next_convolution =
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1, 1, 1, 3 * 63}, {1});
        auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                                        weights_next_convolution,
                                                                                        ngraph::Strides{1, 1},
                                                                                        ngraph::CoordinateDiff{0, 0},
                                                                                        ngraph::CoordinateDiff{0, 1},
                                                                                        ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params_convolution});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeAfterConvOrPoolTestStartMaxPool) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params =
            std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, ngraph::Shape{1, 3, 1, 64});

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(input_params,
                                                                            ngraph::Strides{1, 1},
                                                                            ngraph::Shape{0, 0},
                                                                            ngraph::Shape{0, 1},
                                                                            ngraph::Shape{1, 2});

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 3 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(max_pool_operation, new_shape, true);

        auto weights_next_convolution =
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1, 1, 1, 3 * 63}, {1});
        auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                                        weights_next_convolution,
                                                                                        ngraph::Strides{1, 1},
                                                                                        ngraph::CoordinateDiff{0, 0},
                                                                                        ngraph::CoordinateDiff{0, 1},
                                                                                        ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::InsertTransposeAfterConvOrPool>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params =
            std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, ngraph::Shape{1, 3, 1, 64});

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(input_params,
                                                                            ngraph::Strides{1, 1},
                                                                            ngraph::Shape{0, 0},
                                                                            ngraph::Shape{0, 1},
                                                                            ngraph::Shape{1, 2});

        auto new_shape_out = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 64, 1, 3});
        auto reshape_out_operation =
            std::make_shared<ngraph::opset7::Reshape>(max_pool_operation, new_shape_out, false);

        auto transpose = std::make_shared<ngraph::opset7::Transpose>(
            reshape_out_operation,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2}));

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 3 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(transpose, new_shape, true);

        auto weights_next_convolution =
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1, 1, 1, 3 * 63}, {1});
        auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                                        weights_next_convolution,
                                                                                        ngraph::Strides{1, 1},
                                                                                        ngraph::CoordinateDiff{0, 0},
                                                                                        ngraph::CoordinateDiff{0, 1},
                                                                                        ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);
        reference_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeAfterConvOrPoolTestInputRank3) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_convolution =
            std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, ngraph::Shape{1, 3, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2, 3, 2}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                   weights,
                                                                                   ngraph::Strides{1},
                                                                                   ngraph::CoordinateDiff{0},
                                                                                   ngraph::CoordinateDiff{1},
                                                                                   ngraph::Strides{1});

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(convolution_operation, new_shape, true);

        auto weights_next_convolution =
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1, 1, 63}, {1});
        auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                                        weights_next_convolution,
                                                                                        ngraph::Strides{1},
                                                                                        ngraph::CoordinateDiff{0},
                                                                                        ngraph::CoordinateDiff{1},
                                                                                        ngraph::Strides{1});

        auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);

        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::InsertTransposeAfterConvOrPool>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_convolution =
            std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, ngraph::Shape{1, 3, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2, 3, 2}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                   weights,
                                                                                   ngraph::Strides{1},
                                                                                   ngraph::CoordinateDiff{0},
                                                                                   ngraph::CoordinateDiff{1},
                                                                                   ngraph::Strides{1});

        auto new_shape_out = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 64, 2});
        auto reshape_out_operation =
            std::make_shared<ngraph::opset7::Reshape>(convolution_operation, new_shape_out, false);

        auto transpose = std::make_shared<ngraph::opset7::Transpose>(
            reshape_out_operation,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1}));

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(transpose, new_shape, true);

        auto weights_next_convolution =
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1, 1, 63}, {1});
        auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                                        weights_next_convolution,
                                                                                        ngraph::Strides{1},
                                                                                        ngraph::CoordinateDiff{0},
                                                                                        ngraph::CoordinateDiff{1},
                                                                                        ngraph::Strides{1});

        auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params_convolution});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

std::shared_ptr<ngraph::Function> CreatePoolConvFunction(const ngraph::Shape& input_shape,
                                                         const ngraph::Shape& pool_kernel_shape) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);

    auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(input_params,
                                                                        pool_kernel_shape,
                                                                        ngraph::Shape{0, 0},
                                                                        ngraph::Shape{0, 1},
                                                                        pool_kernel_shape);

    auto pool_out_shape = max_pool_operation->get_output_shape(0);
    ngraph::Shape new_shape = {
        1,
        1,
        1,
        std::accumulate(std::begin(pool_out_shape), std::end(pool_out_shape), size_t{1}, std::multiplies<size_t>())};
    auto new_shape_const = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, new_shape);
    auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(max_pool_operation, new_shape_const, true);

    auto weights_next_convolution = ngraph::opset7::Constant::create(ngraph::element::i64, new_shape, {1});
    auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                                    weights_next_convolution,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::CoordinateDiff{0, 0},
                                                                                    ngraph::CoordinateDiff{0, 1},
                                                                                    ngraph::Strides{1, 1});

    auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

TEST(TransformationTests, InsertTransposeAfterConvOrPoolTest1dOutput) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    func = CreatePoolConvFunction(ngraph::Shape{1, 3, 1, 8}, ngraph::Strides{1, 8});

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::InsertTransposeAfterConvOrPool>();
    m.run_passes(func);
    ASSERT_NO_THROW(check_rt_info(func));

    reference_func = ngraph::clone_function(*func);

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

}  // namespace testing
