// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>

#include "transformations/insert_transpose_before_matmul.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {

namespace {

std::shared_ptr<ngraph::Function> createFunction(const ngraph::PartialShape& input_values,
                                                     const ngraph::Shape& reshape_values,
                                                     const ngraph::Shape& matmul_values) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_values);

    auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{reshape_values.size()}, reshape_values);
    auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);

    auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{matmul_values.size()}, matmul_values);
    auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(reshape_operation, constant);

    auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
}

// ---------------------------------------------------------------------------------------------------------------------

class InsertTransposeBeforeMatmulTestInvalidFixture: public CommonTestUtils::TestsCommon,
                               public ::testing::WithParamInterface<std::tuple<ngraph::PartialShape, ngraph::Shape, ngraph::Shape>> {
public:
    void SetUp() override;
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void InsertTransposeBeforeMatmulTestInvalidFixture::SetUp() {
    ngraph::PartialShape input_shape;
    ngraph::Shape reshape_shape, matmul_shape;
    std::tie(input_shape, reshape_shape, matmul_shape) = this->GetParam();

    function = createFunction(input_shape, reshape_shape, matmul_shape);
    reference_function = createFunction(input_shape, reshape_shape, matmul_shape);
}

// ---------------------------------------------------------------------------------------------------------------------

class InsertTransposeBeforeMatmulTestFixture: public CommonTestUtils::TestsCommon,
                               public ::testing::WithParamInterface<std::tuple<ngraph::PartialShape, ngraph::Shape, ngraph::Shape>> {
public:
    void SetUp() override;
    std::shared_ptr<ngraph::Function> get_initial_function(const ngraph::PartialShape & input_shape,
                                                   const ngraph::Shape & reshape_shape,
                                                   const ngraph::Shape & matmul_shape);
    std::shared_ptr<ngraph::Function> get_reference(const ngraph::PartialShape & input_shape);
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void InsertTransposeBeforeMatmulTestFixture::SetUp() {
    ngraph::PartialShape input_shape;
    ngraph::Shape reshape_shape, matmul_shape;
    std::tie(input_shape, reshape_shape, matmul_shape) = this->GetParam();

    function = get_initial_function(input_shape, reshape_shape, matmul_shape);
    reference_function = get_reference(input_shape);
}

std::shared_ptr<ngraph::Function> InsertTransposeBeforeMatmulTestFixture::get_initial_function(const ngraph::PartialShape & input_shape,
                                                   const ngraph::Shape & reshape_shape,
                                                   const ngraph::Shape & matmul_shape) {
    return createFunction(input_shape, reshape_shape, matmul_shape);
}

std::shared_ptr<ngraph::Function> InsertTransposeBeforeMatmulTestFixture::get_reference(const ngraph::PartialShape & input_shape) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);

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
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
}

// ---------------------------------------------------------------------------------------------------------------------

void execute_test(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<GNAPluginNS::InsertTransposeBeforeMatmul>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST_P(InsertTransposeBeforeMatmulTestFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(InsertTransposeBeforeMatmulTestSuite, InsertTransposeBeforeMatmulTestFixture,
                        ::testing::Values(std::make_tuple(ngraph::PartialShape{2, 8}, ngraph::Shape{8, 2}, ngraph::Shape{2, 1}),
                                          std::make_tuple(ngraph::PartialShape{1, 16}, ngraph::Shape{8, 2}, ngraph::Shape{2, 1})));

TEST_P(InsertTransposeBeforeMatmulTestInvalidFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(InsertTransposeBeforeMatmulTestInvalidSuite, InsertTransposeBeforeMatmulTestInvalidFixture,
                        ::testing::Values(std::make_tuple(ngraph::PartialShape{2, 9}, ngraph::Shape{9, 2}, ngraph::Shape{2, 1}),
                                          std::make_tuple(ngraph::PartialShape{9, 2}, ngraph::Shape{9, 2}, ngraph::Shape{2, 1})));

} // namespace

} // namespace testing
