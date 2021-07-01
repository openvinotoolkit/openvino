// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/split_convolution_with_large_buffer_size.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {

// use constexpr uint32_t bufferMaxSize = 65528 from gna_limitations

TEST(TransformationTests, SplitConvolutionTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 64, 4096, 4096});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 64, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(convolution_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 64, 4096, 4096});

        auto split_node_c1 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{3});
        auto split_node_c2 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({5}), {960, 960, 960, 960, 256});
        auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(input_params,
                                                                            split_node_c1,
                                                                            split_node_c2);

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 64, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});

        ngraph::OutputVector output_conv_operations;
        for (int i = 0; i < 5; ++i) {
            auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(split_node->output(i),
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});
            output_conv_operations.push_back(convolution_operation);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(output_conv_operations, 3);

        auto result = std::make_shared<ngraph::opset7::Result>(concat);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionTestSmallSize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 1, 1, 1});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 1, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(convolution_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 1, 1, 1});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 1, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(convolution_operation);

        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionWithBiasTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 64, 4096, 4096});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 64, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        add_bias);

        auto result = std::make_shared<ngraph::opset7::Result>(add_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithBias>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 64, 4096, 4096});

        auto split_node_c1 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{3});
        auto split_node_c2 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({5}), {960, 960, 960, 960, 256});
        auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(input_params,
                                                                            split_node_c1,
                                                                            split_node_c2);

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 64, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});

        ngraph::OutputVector output_add_operations;
        for (int i = 0; i < 5; ++i) {
            auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(split_node->output(i),
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});
            auto add_bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        add_bias);
            output_add_operations.push_back(add_operation);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(output_add_operations, 3);

        auto result = std::make_shared<ngraph::opset7::Result>(concat);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionWithBiasTestSmallSize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 1, 1, 1});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 1, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        add_bias);

        auto result = std::make_shared<ngraph::opset7::Result>(add_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithBias>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 1, 1, 1});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 1, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        add_bias);

        auto result = std::make_shared<ngraph::opset7::Result>(add_operation);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

// Variant Convolution -> FakeQuantize

TEST(TransformationTests, SplitConvolutionWithFqTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 64, 4096, 4096});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 64, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(convolution_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithFq>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 64, 4096, 4096});

        auto split_node_c1 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{3});
        auto split_node_c2 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({5}), {960, 960, 960, 960, 256});
        auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(input_params,
                                                                            split_node_c1,
                                                                            split_node_c2);

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 64, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});

        ngraph::OutputVector output_fq_operations;
        for (int i = 0; i < 5; ++i) {
            auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(split_node->output(i),
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(convolution_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);
            output_fq_operations.push_back(fake_quantize_op);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(output_fq_operations, 3);

        auto result = std::make_shared<ngraph::opset7::Result>(concat);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionWithFqTestSmallSize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 1, 1, 1});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 1, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(convolution_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithFq>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 1, 1, 1});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 1, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(convolution_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

// Variant Convolution -> Add -> FakeQuantize

TEST(TransformationTests, SplitConvolutionWithFqAddTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 64, 4096, 4096});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 64, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        add_bias);

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(add_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithFq>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 64, 4096, 4096});

        auto split_node_c1 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{3});
        auto split_node_c2 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({5}), {960, 960, 960, 960, 256});
        auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(input_params,
                                                                            split_node_c1,
                                                                            split_node_c2);

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 64, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});

        ngraph::OutputVector output_fq_operations;
        for (int i = 0; i < 5; ++i) {
            auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(split_node->output(i),
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});
            auto add_bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        add_bias);

            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(add_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);
            output_fq_operations.push_back(fake_quantize_op);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(output_fq_operations, 3);

        auto result = std::make_shared<ngraph::opset7::Result>(concat);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionWithFqAddTestSmallSize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 1, 1, 1});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 1, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        add_bias);

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(add_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithFq>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 1, 1, 1});

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{1, 1, 1, 1}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{1, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        add_bias);

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(add_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace testing
