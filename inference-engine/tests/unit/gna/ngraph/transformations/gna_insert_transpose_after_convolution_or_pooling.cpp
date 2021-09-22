// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/insert_transpose_after_convolution_or_pooling.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {

TEST(TransformationTests, InsertTransposeAfterConvOrPoolTestStartConvolution) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                        ngraph::Shape{1, 3, 1, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{3, 3, 1, 2}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 1},
                                                                  ngraph::Strides{1, 1});

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 3 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(convolution_operation, new_shape, true);

        auto weights_next_convolution = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{1, 1, 1, 3 * 63}, {1});
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
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeAfterConvOrPool>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                        ngraph::Shape{1, 3, 1, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{3, 3, 1, 2}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 1},
                                                                  ngraph::Strides{1, 1});

        auto new_shape_out = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 64, 1, 3});
        auto reshape_out_operation = std::make_shared<ngraph::opset7::Reshape>(convolution_operation, new_shape_out, false);

        auto transpose = std::make_shared<ngraph::opset7::Transpose>(reshape_out_operation,
                                                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2}));

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 3 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(transpose, new_shape, true);

        auto weights_next_convolution = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{1, 1, 1, 3 * 63}, {1});
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

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeAfterConvOrPoolTestStartMaxPool) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                        ngraph::Shape{1, 3, 1, 64});

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(input_params,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::Shape{0, 0},
                                                                                    ngraph::Shape{0, 1},
                                                                                    ngraph::Shape{1, 2});

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 3 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(max_pool_operation, new_shape, true);

        auto weights_next_convolution = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{1, 1, 1, 3 * 63}, {1});
        auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                  weights_next_convolution,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 1},
                                                                  ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeAfterConvOrPool>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                        ngraph::Shape{1, 3, 1, 64});

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(input_params,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::Shape{0, 0},
                                                                                    ngraph::Shape{0, 1},
                                                                                    ngraph::Shape{1, 2});

        auto new_shape_out = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 64, 1, 3});
        auto reshape_out_operation = std::make_shared<ngraph::opset7::Reshape>(max_pool_operation, new_shape_out, false);

        auto transpose = std::make_shared<ngraph::opset7::Transpose>(reshape_out_operation,
                                                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2}));

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 3 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(transpose, new_shape, true);

        auto weights_next_convolution = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{1, 1, 1, 3 * 63}, {1});
        auto next_convolution_operation = std::make_shared<ngraph::opset7::Convolution>(reshape_operation,
                                                                  weights_next_convolution,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 1},
                                                                  ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(next_convolution_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, InsertTransposeAfterConvOrPoolTestInputRank3) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                        ngraph::Shape{1, 3, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{2, 3, 2}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1},
                                                                  ngraph::CoordinateDiff{0},
                                                                  ngraph::CoordinateDiff{1},
                                                                  ngraph::Strides{1});

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(convolution_operation, new_shape, true);

        auto weights_next_convolution = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{1, 1, 63}, {1});
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
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertTransposeAfterConvOrPool>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                        ngraph::Shape{1, 3, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{2, 3, 2}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1},
                                                                  ngraph::CoordinateDiff{0},
                                                                  ngraph::CoordinateDiff{1},
                                                                  ngraph::Strides{1});

        auto new_shape_out = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 64, 2});
        auto reshape_out_operation = std::make_shared<ngraph::opset7::Reshape>(convolution_operation, new_shape_out, false);

        auto transpose = std::make_shared<ngraph::opset7::Transpose>(reshape_out_operation,
                                                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1}));

        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(transpose, new_shape, true);

        auto weights_next_convolution = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                        ngraph::Shape{1, 1, 63}, {1});
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

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace testing
