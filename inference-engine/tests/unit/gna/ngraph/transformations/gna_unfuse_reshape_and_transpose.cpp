// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/unfuse_reshape_and_transpose.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/serialize.hpp>

namespace testing {
namespace {

static std::shared_ptr<ngraph::Function> createFunction(const ngraph::Shape& conv_input_shape,
                                                        const ngraph::Shape& conv_filter_shape,
                                                        bool with_bias,
                                                        bool with_pool,
                                                        bool single_reshape_before,
                                                        bool single_reshape_after) {
    size_t total_in = std::accumulate(std::begin(conv_input_shape), std::end(conv_input_shape), 1, std::multiplies<int>());
    auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, total_in});
    std::shared_ptr<ngraph::Node> last_node;
    if (single_reshape_before) {
        auto reshape_in_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, conv_input_shape);
        auto reshape_in = std::make_shared<ngraph::opset8::Reshape>(input, reshape_in_const, false);
        last_node = reshape_in;
    } else {
        auto reshape_in_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4},
            ngraph::Shape{conv_input_shape[0], conv_input_shape[2], conv_input_shape[3], conv_input_shape[1]});
        auto reshape_in = std::make_shared<ngraph::opset8::Reshape>(input, reshape_in_const, false);
        auto transpose_in_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
        auto transpose_in = std::make_shared<ngraph::opset8::Transpose>(reshape_in, transpose_in_const);
        last_node = transpose_in;
    }
    auto conv_weights = ngraph::opset8::Constant::create(ngraph::element::f32, conv_filter_shape, {1});
    auto conv = std::make_shared<ngraph::opset8::Convolution>(last_node,
                                                              conv_weights,
                                                              ngraph::Strides{1, 1},
                                                              ngraph::CoordinateDiff{0, 0},
                                                              ngraph::CoordinateDiff{0, 0},
                                                              ngraph::Strides{1, 1});
    last_node = conv;
    auto conv_output_shape = conv->get_output_shape(0);
    size_t total_out = std::accumulate(std::begin(conv_output_shape), std::end(conv_output_shape), 1, std::multiplies<int>());
    if (with_bias) {
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, conv_output_shape.at(1), 1, 1}, {1});
        auto add = std::make_shared<ngraph::opset8::Add>(conv, add_const);
        last_node = add;
    }
    if (with_pool) {
        auto pool = std::make_shared<ngraph::opset7::MaxPool>(last_node,
            ngraph::Strides{1, 1}, ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, ngraph::Shape{1, 1});
        last_node = pool;
    }
    auto reshape_out_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, total_out});
    if (!single_reshape_after) {
        auto transpose_out_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
        auto transpose_out = std::make_shared<ngraph::opset8::Transpose>(last_node, transpose_out_const);
        last_node = transpose_out;
    }
    auto reshape_out = std::make_shared<ngraph::opset8::Reshape>(last_node, reshape_out_const, false);

    auto result = std::make_shared<ngraph::opset8::Result>(reshape_out);
    auto func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input});

    return func;
}

typedef std::tuple<
        ngraph::Shape,                      // Convolution input shape
        ngraph::Shape,                      // Convolution kernel
        bool,                               // with bias
        bool,                               // with pooling
        bool,                               // expect unfuse before
        bool                                // expect unfuse after
> UnfuseReshapeAndTransposeParams;

class UnfuseReshapeAndTransposeTestSuiteFixture: public CommonTestUtils::TestsCommon,
                               public ::testing::WithParamInterface<UnfuseReshapeAndTransposeParams> {
public:
    void SetUp() override;
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void UnfuseReshapeAndTransposeTestSuiteFixture::SetUp() {
    ngraph::Shape conv_input_shape;
    ngraph::Shape conv_filter_shape;
    bool with_bias;
    bool with_pool;
    bool replace_before;
    bool replace_after;
    std::tie(conv_input_shape, conv_filter_shape, with_bias, with_pool, replace_before, replace_after) = this->GetParam();
    function = createFunction(conv_input_shape, conv_filter_shape, with_bias, with_pool, true, true);
    reference_function = createFunction(conv_input_shape, conv_filter_shape, with_bias, with_pool, !replace_before, !replace_after);
}

void execute_test(std::shared_ptr<ngraph::Function> function,
                  std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<GNAPluginNS::Unfuse2dto4dReshapeAndTranspose>();
    manager.register_pass<GNAPluginNS::Unfuse4dto2dReshapeAndTranspose>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST_P(UnfuseReshapeAndTransposeTestSuiteFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(UnfuseReshapeAndTransposeTestSuite, UnfuseReshapeAndTransposeTestSuiteFixture,
                         ::testing::Values(std::make_tuple(ngraph::Shape{1, 1, 1, 168}, ngraph::Shape{12, 1, 1, 8}, false, false, true, false),
                                           std::make_tuple(ngraph::Shape{1, 1, 1, 640}, ngraph::Shape{256, 1, 1, 512}, false, false, true, false),
                                           std::make_tuple(ngraph::Shape{1, 1, 1, 1024}, ngraph::Shape{256, 1, 1, 512}, false, false, true, false),
                                           std::make_tuple(ngraph::Shape{1, 1, 33, 32}, ngraph::Shape{128, 1, 33, 9}, false, false, true, false),
                                           std::make_tuple(ngraph::Shape{1, 1, 11, 13}, ngraph::Shape{128, 1, 11, 9}, false, false, true, false),
                                           std::make_tuple(ngraph::Shape{1, 1, 33, 23}, ngraph::Shape{128, 1, 11, 5}, false, false, true, false),
                                           std::make_tuple(ngraph::Shape{1, 1, 33, 32}, ngraph::Shape{1, 1, 33, 9}, false, false, true, true),
                                           std::make_tuple(ngraph::Shape{1, 1, 1, 1024}, ngraph::Shape{256, 1, 1, 1024}, false, false, true, true),
                                           std::make_tuple(ngraph::Shape{1, 1, 33, 32}, ngraph::Shape{1, 1, 33, 9}, true, false, true, true),
                                           std::make_tuple(ngraph::Shape{1, 1, 1, 1024}, ngraph::Shape{256, 1, 1, 1024}, true, false, true, true),
                                           std::make_tuple(ngraph::Shape{1, 1, 33, 32}, ngraph::Shape{1, 1, 33, 9}, false, true, true, true),
                                           std::make_tuple(ngraph::Shape{1, 1, 1, 1024}, ngraph::Shape{256, 1, 1, 1024}, false, true, true, true),
                                           std::make_tuple(ngraph::Shape{1, 1, 33, 32}, ngraph::Shape{1, 1, 33, 9}, true, true, true, true),
                                           std::make_tuple(ngraph::Shape{1, 1, 1, 1024}, ngraph::Shape{256, 1, 1, 1024}, true, true, true, true)));

} // namespace
} // namespace testing
