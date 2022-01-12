// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/reorder_activation_and_pooling.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {

namespace {

class IActivationNodeFactory {
public:
    virtual ~IActivationNodeFactory() = default;
    virtual std::shared_ptr<ngraph::Node> createNode(const ngraph::Output<ngraph::Node>& in) = 0;
};

template <typename ActivationT>
class ActivationNodeFactory : public IActivationNodeFactory {
public:
    ActivationNodeFactory() = default;
    std::shared_ptr<ngraph::Node> createNode(const ngraph::Output<ngraph::Node>& operation_before) override {
        return std::make_shared<ActivationT>(operation_before);
    }
private:
    ActivationNodeFactory(const ActivationNodeFactory&) = delete;
    ActivationNodeFactory& operator=(const ActivationNodeFactory& ) = delete;
};

template <>
class ActivationNodeFactory <ngraph::opset7::Clamp> : public IActivationNodeFactory {
public:
    ActivationNodeFactory(const double min, const double max) : min_(min), max_(max) {}
    std::shared_ptr<ngraph::Node> createNode(const ngraph::Output<ngraph::Node>& operation_before) override {
        return std::make_shared<ngraph::opset7::Clamp>(operation_before, min_, max_);
    }
private:
    ActivationNodeFactory(const ActivationNodeFactory&) = delete;
    ActivationNodeFactory& operator=(const ActivationNodeFactory& ) = delete;
private:
    const double min_;
    const double max_;
};

using ActivationFactoryPtr = std::shared_ptr<IActivationNodeFactory>;

template <typename ActivationT, typename ... Args>
ActivationFactoryPtr createActivationFactory(Args&& ... args) {
    return std::make_shared<ActivationNodeFactory<ActivationT>>(std::forward<Args>(args) ...);
}

// ----------------------------------------------------------------------------------------------------------------------

/* Variants:
    Convolution -> Add -> Activation -> MaxPool
    Convolution -> Activation -> MaxPool
 */

typedef std::tuple<
    ActivationFactoryPtr,                // activation Node factory
    bool                                 // do we need to create ngraph::opset7::Add Node or not
> ConvolutionActivationPoolTestOptions;

class ConvolutionActivationPoolTestFixture : public CommonTestUtils::TestsCommon,
                                             public testing::WithParamInterface<ConvolutionActivationPoolTestOptions> {
public:
    void SetUp() override;
    std::shared_ptr<ngraph::Function> get_initial_function(ActivationFactoryPtr activation_factory,
                                                           bool isAddNodeNeeded);
    std::shared_ptr<ngraph::Function> get_reference(ActivationFactoryPtr activation_factory,
                                                    bool isAddNodeNeeded);
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void ConvolutionActivationPoolTestFixture::SetUp() {
    ActivationFactoryPtr activation_factory;
    bool isAddNodeNeeded = false;
    std::tie(activation_factory, isAddNodeNeeded) = GetParam();

    function = get_initial_function(activation_factory, isAddNodeNeeded);
    reference_function = get_reference(activation_factory, isAddNodeNeeded);
}

std::shared_ptr<ngraph::Function> ConvolutionActivationPoolTestFixture::get_initial_function(ActivationFactoryPtr activation_factory,
                                                                                             bool isAddNodeNeeded) {
    auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                                ngraph::Shape{1, 3, 64, 64});
    auto input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

    auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                    ngraph::Shape{3, 3, 1, 1}, {1});
    auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                 ngraph::Shape{3, 1, 1}, {1});
    auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                               weights,
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::Strides{1, 1});

    std::shared_ptr<ngraph::op::Op> last_operation = convolution_operation;
    if (isAddNodeNeeded) {
        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                   input_params_add);
        last_operation = add_operation;
    }
    auto activation = activation_factory->createNode(last_operation);

    auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(activation,
                                                                        ngraph::Strides{1, 1},
                                                                        ngraph::Shape{1, 1},
                                                                        ngraph::Shape{1, 1},
                                                                        ngraph::Shape{1, 1});

    auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                            ngraph::ParameterVector{input_params_convolution,
                                                                    input_params_add});
}

std::shared_ptr<ngraph::Function> ConvolutionActivationPoolTestFixture::get_reference(ActivationFactoryPtr activation_factory,
                                                                                      bool isAddNodeNeeded) {
    auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                                ngraph::Shape{1, 3, 64, 64});

    auto input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

    auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                    ngraph::Shape{3, 3, 1, 1}, {1});
    auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                 ngraph::Shape{3, 1, 1}, {1});
    auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                               weights,
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::Strides{1, 1});

    std::shared_ptr<ngraph::op::Op> last_operation = convolution_operation;
    if (isAddNodeNeeded) {
        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                   input_params_convolution);
        last_operation = add_operation;
    }

    auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(last_operation,
                                                                        ngraph::Strides{1, 1},
                                                                        ngraph::Shape{1, 1},
                                                                        ngraph::Shape{1, 1},
                                                                        ngraph::Shape{1, 1});

    auto activation = activation_factory->createNode(max_pool_operation);

    auto result = std::make_shared<ngraph::opset7::Result>(activation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution,
                                                                          input_params_add});
}

void execute_test(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST_P(ConvolutionActivationPoolTestFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

const std::vector<ActivationFactoryPtr> activationFactories = {
    createActivationFactory<ngraph::opset7::Relu>(),
    createActivationFactory<ngraph::opset7::Sigmoid>(),
    createActivationFactory<ngraph::opset7::Tanh>(),
    createActivationFactory<ngraph::opset7::Abs>(),
    createActivationFactory<ngraph::opset7::Log>(),
    createActivationFactory<ngraph::opset7::Exp>(),
    createActivationFactory<ngraph::opset7::Sign>(),
    createActivationFactory<ngraph::opset7::Clamp>(0.1, 0.2)
};

INSTANTIATE_TEST_SUITE_P(ConvolutionActivationPoolTestSuite, ConvolutionActivationPoolTestFixture,
                         ::testing::Combine(::testing::ValuesIn(activationFactories),
                                            ::testing::ValuesIn(std::vector<bool>{true, false})));

//-----------------------------------------------------------------------------------------------------------

// Variant Convolution -> FakeQuantize -> MaxPool : ConvFqMp

TEST(TransformationTests, ReorderActivationAndPoolingTestConvFqMp) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                                    ngraph::Shape{1, 3, 64, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
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

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(fake_quantize_op,
                                                                            ngraph::Strides{1, 1},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                                    ngraph::Shape{1, 3, 64, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                   weights,
                                                                                   ngraph::Strides{1, 1},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::Strides{1, 1});

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(convolution_operation,
                                                                            ngraph::Strides{1, 1},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{1, 1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(max_pool_operation, input_low,
                                                                               input_high, output_low,
                                                                               output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input_params_convolution});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

// Variant Convolution -> Add -> FakeQuantize -> MaxPool : ConvAddFqMp

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddFqMp) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                                    ngraph::Shape{1, 3, 64, 64});

        auto input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                            ngraph::Shape{1, 3, 64, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                   weights,
                                                                                   ngraph::Strides{1, 1},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::Strides{1, 1});

        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                   input_params_add);

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(add_operation, input_low,
                                                                               input_high, output_low,
                                                                               output_high, 11);

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(fake_quantize_op,
                                                                            ngraph::Strides{1, 1},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution, input_params_add});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                                    ngraph::Shape{1, 3, 64, 64});

        auto input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                            ngraph::Shape{1, 3, 64, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                   weights,
                                                                                   ngraph::Strides{1, 1},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::Strides{1, 1});

        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                   input_params_add);

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(add_operation,
                                                                            ngraph::Strides{1, 1},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{1, 1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(max_pool_operation, input_low,
                                                                               input_high, output_low,
                                                                               output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution, input_params_add});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace

} // namespace testing
