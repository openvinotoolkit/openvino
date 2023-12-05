// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/unfuse_reshape_and_transpose.hpp"

namespace testing {
namespace {

class IActivationFactory {
public:
    virtual ~IActivationFactory() = default;
    virtual std::shared_ptr<ngraph::Node> createNode(const ngraph::Output<ngraph::Node>& in) = 0;
};

template <typename T>
class ActivationFactory : public IActivationFactory {
public:
    ActivationFactory() = default;
    std::shared_ptr<ngraph::Node> createNode(const ngraph::Output<ngraph::Node>& operation_before) override {
        return std::make_shared<T>(operation_before);
    }

private:
    ActivationFactory(const ActivationFactory&) = delete;
    ActivationFactory& operator=(const ActivationFactory&) = delete;
};

template <>
class ActivationFactory<ngraph::opset8::Clamp> : public IActivationFactory {
public:
    ActivationFactory(const double min, const double max) : min_(min), max_(max) {}
    std::shared_ptr<ngraph::Node> createNode(const ngraph::Output<ngraph::Node>& operation_before) override {
        return std::make_shared<ngraph::opset8::Clamp>(operation_before, min_, max_);
    }

private:
    ActivationFactory(const ActivationFactory&) = delete;
    ActivationFactory& operator=(const ActivationFactory&) = delete;

private:
    const double min_;
    const double max_;
};

using ActivationFactoryPtr = std::shared_ptr<IActivationFactory>;

template <typename T, typename... Args>
ActivationFactoryPtr createActivationFactory(Args&&... args) {
    return std::make_shared<ActivationFactory<T>>(std::forward<Args>(args)...);
}

static std::shared_ptr<ngraph::Function> createFunction(const ngraph::Shape& conv_input_shape,
                                                        const ngraph::Shape& conv_filter_shape,
                                                        bool with_bias,
                                                        bool with_pool,
                                                        ActivationFactoryPtr activation_factory,
                                                        bool with_fq,
                                                        bool single_reshape_before,
                                                        bool single_reshape_after,
                                                        bool single_batch) {
    size_t total_in =
        std::accumulate(std::begin(conv_input_shape), std::end(conv_input_shape), 1, std::multiplies<int>());
    auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, total_in});
    std::shared_ptr<ngraph::Node> last_node, last_const;
    auto add_fake_quantize = [&](const std::shared_ptr<ngraph::Node>& node) {
        auto input_low = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {5});
        auto output_low = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        return std::make_shared<ngraph::opset8::FakeQuantize>(node, input_low, input_high, output_low, output_high, 11);
    };
    if (single_reshape_before) {
        auto reshape_in_const =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, conv_input_shape);
        auto reshape_in = std::make_shared<ngraph::opset8::Reshape>(input, reshape_in_const, false);
        last_node = reshape_in;
    } else {
        auto reshape_in_const = ngraph::opset8::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{4},
            ngraph::Shape{conv_input_shape[0], conv_input_shape[2], conv_input_shape[3], conv_input_shape[1]});
        auto reshape_in = std::make_shared<ngraph::opset8::Reshape>(input, reshape_in_const, false);
        auto transpose_in_const =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
        auto transpose_in = std::make_shared<ngraph::opset8::Transpose>(reshape_in, transpose_in_const);
        last_node = transpose_in;
    }
    auto conv_weights = ngraph::opset8::Constant::create(ngraph::element::f32, conv_filter_shape, {1});
    last_const = conv_weights;
    if (with_fq) {
        auto conv_input_fq = add_fake_quantize(last_node);
        last_node = conv_input_fq;
        auto conv_weights_fq = add_fake_quantize(conv_weights);
        last_const = conv_weights_fq;
    }
    auto conv = std::make_shared<ngraph::opset8::Convolution>(last_node,
                                                              last_const,
                                                              ngraph::Strides{1, 1},
                                                              ngraph::CoordinateDiff{0, 0},
                                                              ngraph::CoordinateDiff{0, 0},
                                                              ngraph::Strides{1, 1});
    last_node = conv;
    auto conv_output_shape = conv->get_output_shape(0);
    size_t total_out =
        std::accumulate(std::begin(conv_output_shape), std::end(conv_output_shape), 1, std::multiplies<int>());
    if (with_bias) {
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32,
                                                          ngraph::Shape{1, conv_output_shape.at(1), 1, 1},
                                                          {1});
        auto add = std::make_shared<ngraph::opset8::Add>(conv, add_const);
        last_node = add;
    }
    if (with_fq) {
        auto conv_bias_fq = add_fake_quantize(last_node);
        last_node = conv_bias_fq;
    }
    if (with_pool) {
        auto pool = std::make_shared<ngraph::opset7::MaxPool>(last_node,
                                                              ngraph::Strides{1, 1},
                                                              ngraph::Shape{0, 0},
                                                              ngraph::Shape{0, 0},
                                                              ngraph::Shape{1, 1});
        last_node = pool;
    }
    if (activation_factory) {
        if (with_fq) {
            auto act_fq_in = add_fake_quantize(last_node);
            last_node = act_fq_in;
        }
        auto act = activation_factory->createNode(last_node);
        last_node = act;
        if (with_fq) {
            auto act_fq_out = add_fake_quantize(last_node);
            last_node = act_fq_out;
        }
    }
    auto out_shape = single_batch ? ngraph::Shape{1, total_out} : ngraph::Shape{total_out, 1};
    auto reshape_out_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, out_shape);
    if (!single_reshape_after) {
        auto transpose_out_const =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
        auto transpose_out = std::make_shared<ngraph::opset8::Transpose>(last_node, transpose_out_const);
        last_node = transpose_out;
    }
    auto reshape_out = std::make_shared<ngraph::opset8::Reshape>(last_node, reshape_out_const, false);

    auto result = std::make_shared<ngraph::opset8::Result>(reshape_out);
    auto func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input});

    return func;
}

typedef std::tuple<std::tuple<ngraph::Shape, ngraph::Shape, bool, bool>,
                   bool,                  // with bias
                   bool,                  // with pooling
                   ActivationFactoryPtr,  // with activation
                   bool,                  // with fq
                   bool                   // out batch is 1 or not
                   >
    UnfuseReshapeAndTransposeParams;

class UnfuseReshapeAndTransposeTestSuiteFixture
    : public ov::test::TestsCommon,
      public ::testing::WithParamInterface<UnfuseReshapeAndTransposeParams> {
public:
    void SetUp() override;

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void UnfuseReshapeAndTransposeTestSuiteFixture::SetUp() {
    std::tuple<ngraph::Shape, ngraph::Shape, bool, bool> conv_data;
    bool with_bias;
    bool with_pool;
    bool with_fq;
    bool single_batch;
    ActivationFactoryPtr af;
    std::tie(conv_data, with_bias, with_pool, af, with_fq, single_batch) = this->GetParam();
    ngraph::Shape conv_input_shape;
    ngraph::Shape conv_filter_shape;
    bool replace_before;
    bool replace_after;
    std::tie(conv_input_shape, conv_filter_shape, replace_before, replace_after) = conv_data;
    function = createFunction(conv_input_shape,
                              conv_filter_shape,
                              with_bias,
                              with_pool,
                              af,
                              with_fq,
                              true,
                              true,
                              single_batch);
    reference_function = createFunction(conv_input_shape,
                                        conv_filter_shape,
                                        with_bias,
                                        with_pool,
                                        af,
                                        with_fq,
                                        !replace_before,
                                        !replace_after,
                                        single_batch);
}

void execute_test(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::Unfuse2dto4dReshapeAndTranspose>();
    manager.register_pass<ov::intel_gna::pass::Unfuse4dto2dReshapeAndTranspose>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST_P(UnfuseReshapeAndTransposeTestSuiteFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

const std::vector<ActivationFactoryPtr> activationFactories = {
    nullptr,
    createActivationFactory<ngraph::opset8::Relu>(),
    createActivationFactory<ngraph::opset8::Sigmoid>(),
    createActivationFactory<ngraph::opset8::Tanh>(),
    createActivationFactory<ngraph::opset8::Abs>(),
    createActivationFactory<ngraph::opset8::Log>(),
    createActivationFactory<ngraph::opset8::Exp>(),
    createActivationFactory<ngraph::opset8::Sign>(),
    createActivationFactory<ngraph::opset8::Clamp>(0.1, 0.2)};

INSTANTIATE_TEST_SUITE_P(
    UnfuseReshapeAndTransposeTestSuite,
    UnfuseReshapeAndTransposeTestSuiteFixture,
    ::testing::Combine(::testing::ValuesIn(std::vector<std::tuple<ngraph::Shape, ngraph::Shape, bool, bool>>{
                           {ngraph::Shape{1, 1, 1, 168}, ngraph::Shape{12, 1, 1, 8}, true, false},
                           {ngraph::Shape{1, 1, 1, 640}, ngraph::Shape{256, 1, 1, 512}, true, false},
                           {ngraph::Shape{1, 1, 1, 1024}, ngraph::Shape{256, 1, 1, 512}, true, false},
                           {ngraph::Shape{1, 1, 33, 32}, ngraph::Shape{128, 1, 33, 9}, true, false},
                           {ngraph::Shape{1, 1, 11, 13}, ngraph::Shape{128, 1, 11, 9}, true, false},
                           {ngraph::Shape{1, 1, 33, 23}, ngraph::Shape{128, 1, 11, 5}, true, false},
                           {ngraph::Shape{1, 1, 33, 32}, ngraph::Shape{1, 1, 33, 9}, true, true},
                           {ngraph::Shape{1, 1, 1, 1024}, ngraph::Shape{256, 1, 1, 1024}, true, true},
                           {ngraph::Shape{1, 1, 33, 32}, ngraph::Shape{1, 1, 33, 9}, true, true}}),
                       ::testing::ValuesIn(std::vector<bool>{true, false}),    // with bias
                       ::testing::ValuesIn(std::vector<bool>{true, false}),    // with max pool
                       ::testing::ValuesIn(activationFactories),               // with activation
                       ::testing::ValuesIn(std::vector<bool>{true, false}),    // with fq
                       ::testing::ValuesIn(std::vector<bool>{true, false})));  // out batch is 1

}  // namespace
}  // namespace testing
