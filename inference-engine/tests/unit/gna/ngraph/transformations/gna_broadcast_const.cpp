// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/broadcast_const.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include "legacy/ngraph_ops/eltwise.hpp"

namespace testing {

std::shared_ptr<ngraph::opset8::FakeQuantize> createFakeQuantizeNode(std::shared_ptr<ngraph::op::Op> parent_node) {
    auto input_low = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {-0.5});
    auto input_high = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {0.5});
    auto output_low = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {-0.5});
    auto output_high = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {0.5});
    return std::make_shared<ngraph::opset8::FakeQuantize>(parent_node, input_low,
                                                          input_high, output_low,
                                                          output_high, 0);
}

TEST(TransformationTests, BroadcastConstTestFakeQuantize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{3, 2};

    auto create_graph = [](const ngraph::Shape& data_shape, const ngraph::Shape& const_shape_dims,
                           const ngraph::Shape& const_shape_values) {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);

        auto constant = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                         ngraph::Shape{const_shape_dims}, const_shape_values);

        auto fakeQuantize1 = createFakeQuantizeNode(input_params);
        auto fakeQuantize2 = createFakeQuantizeNode(constant);

        auto add = std::make_shared<ngraph::opset8::Add>(fakeQuantize1, fakeQuantize2);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    };

    {
        func = create_graph(data_shape, {2}, {1, 2});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<GNAPluginNS::BroadcastConst>();
        manager.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = create_graph(data_shape, {3, 2}, {1, 2, 1, 2, 1, 2});

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, BroadcastConstTestFakeQuantizeSwapFq) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{3, 2};

    auto create_graph = [](const ngraph::Shape& data_shape, const ngraph::Shape& const_shape_dims,
                           const ngraph::Shape& const_shape_values) {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);

        auto constant = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                         ngraph::Shape{const_shape_dims}, const_shape_values);

        auto fakeQuantize1 = createFakeQuantizeNode(input_params);
        auto fakeQuantize2 = createFakeQuantizeNode(constant);

        auto add = std::make_shared<ngraph::opset8::Add>(fakeQuantize2, fakeQuantize1);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    };

    {
        func = create_graph(data_shape, {2}, {1, 2});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<GNAPluginNS::BroadcastConst>();
        manager.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = create_graph(data_shape, {3, 2}, {1, 2, 1, 2, 1, 2});

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, BroadcastConstTestFakeQuantizeEltwise) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{3, 2};

    auto create_graph = [](const ngraph::Shape& data_shape, const ngraph::Shape& const_shape_dims,
                           const ngraph::Shape& const_shape_values) {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);

        auto constant = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                         ngraph::Shape{const_shape_dims}, const_shape_values);

        auto fakeQuantize1 = createFakeQuantizeNode(input_params);
        auto fakeQuantize2 = createFakeQuantizeNode(constant);

        auto add = std::make_shared<ngraph::op::Eltwise>(fakeQuantize1, fakeQuantize2, ELTWISE_TYPE::Sum);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    };

    {
        func = create_graph(data_shape, {2}, {1, 2});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<GNAPluginNS::BroadcastConst>();
        manager.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = create_graph(data_shape, {3, 2}, {1, 2, 1, 2, 1, 2});

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, BroadcastConstTestMain) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{3, 2};

    auto create_graph = [](const ngraph::Shape& data_shape, const ngraph::Shape& const_shape_dims,
                           const ngraph::Shape& const_shape_values) {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);

        auto constant = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                         const_shape_dims, const_shape_values);

        auto add = std::make_shared<ngraph::opset8::Add>(constant, input_params);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    };

    {
        func = create_graph(data_shape, {2}, {1, 2});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<GNAPluginNS::BroadcastConst>();
        manager.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = create_graph(data_shape, {3, 2}, {1, 2, 1, 2, 1, 2});

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, BroadcastConstTestMainSwapInputs) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{3, 2};

    auto create_graph = [](const ngraph::Shape& data_shape, const ngraph::Shape& const_shape_dims,
                           const ngraph::Shape& const_shape_values) {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);

        auto constant = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                         const_shape_dims, const_shape_values);

        auto add = std::make_shared<ngraph::opset8::Add>(input_params, constant);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    };

    {
        func = create_graph(data_shape, {2}, {1, 2});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<GNAPluginNS::BroadcastConst>();
        manager.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = create_graph(data_shape, {3, 2}, {1, 2, 1, 2, 1, 2});

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

#ifdef GNA_LEGACY
TEST(TransformationTests, BroadcastConstTestConvolution) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{2, 2, 1, 1};

    {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);

        auto constant = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                         ngraph::Shape{1, 2, 1, 1}, {1});

        auto kernel = ngraph::opset8::Constant::create(ngraph::element::f32,
                                                       {2, 2, 1, 1}, {1});

        auto convolution = std::make_shared<ngraph::opset8::Convolution>(input_params,
                                                         kernel,
                                                         ngraph::Strides{1, 1},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::Strides{1, 1});

        auto add = std::make_shared<ngraph::opset8::Add>(constant, convolution);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<GNAPluginNS::BroadcastConst>();
        manager.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = ngraph::clone_function(*func);

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, BroadcastConstTestConvolutionSwapInputs) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{2, 2, 1, 1};

    {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);

        auto constant = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                         ngraph::Shape{1, 2, 1, 1}, {1});

        auto kernel = ngraph::opset8::Constant::create(ngraph::element::f32,
                                                       {2, 2, 1, 1}, {1});

        auto convolution = std::make_shared<ngraph::opset8::Convolution>(input_params,
                                                         kernel,
                                                         ngraph::Strides{1, 1},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::Strides{1, 1});

        auto add = std::make_shared<ngraph::opset8::Add>(constant, convolution);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<GNAPluginNS::BroadcastConst>();
        manager.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = ngraph::clone_function(*func);

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}
#endif

} // namespace testing
