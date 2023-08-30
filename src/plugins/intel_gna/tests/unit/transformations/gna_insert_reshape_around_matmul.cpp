// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <numeric>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/insert_reshape_around_matmul.hpp"

template <bool ADD = false,
          bool ADD_FIRST_INPUT_NOT_CONSTANT = false,
          bool ADD_FULL_DIM = false,
          bool FQ = false,
          bool TRANSPOSE = false>
struct InsertReshapeAroundMatmulTest {
    static std::shared_ptr<ngraph::Node> CreateAdd(std::shared_ptr<ngraph::Node> input,
                                                   const ngraph::Shape& constant_shape) {
        std::vector<size_t> data(ngraph::shape_size(constant_shape));
        std::iota(std::begin(data), std::end(data), 1);
        auto constant = ngraph::opset8::Constant::create(ngraph::element::i64, constant_shape, data);
        return std::make_shared<ngraph::opset8::Add>(input, constant);
    }

    static std::shared_ptr<ngraph::Node> CreateMatmul(std::shared_ptr<ngraph::Node> input,
                                                      const ngraph::Shape& matmul_constant_shape,
                                                      const ngraph::Shape& permutation_shape) {
        std::vector<size_t> data(ngraph::shape_size(matmul_constant_shape));
        std::iota(std::begin(data), std::end(data), 1);
        auto constant = ngraph::opset8::Constant::create(ngraph::element::i64, matmul_constant_shape, data);
        std::shared_ptr<ngraph::Node> node;
        node = std::make_shared<ngraph::opset8::MatMul>(input, constant);

        if (ADD) {
            std::vector<size_t> add_constant_shape(2, 1);
            auto matmul_shape = node->get_output_shape(0);
            data.resize(ngraph::shape_size(matmul_shape));
            std::iota(std::begin(data), std::end(data), 1);

            if (ADD_FULL_DIM) {
                add_constant_shape.resize(matmul_shape.size(), 1);
                std::copy(matmul_shape.begin(), matmul_shape.end(), add_constant_shape.begin());
            } else {
                std::copy_if(matmul_shape.begin(), matmul_shape.end(), add_constant_shape.begin(), [](size_t e) {
                    return e > 1;
                });
            }

            auto constant_add =
                ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{add_constant_shape}, data);
            if (ADD_FIRST_INPUT_NOT_CONSTANT) {
                node = std::make_shared<ngraph::opset8::Add>(node, constant_add);
            } else {
                node = std::make_shared<ngraph::opset8::Add>(constant_add, node);
            }
        }

        if (FQ) {
            node = std::make_shared<ngraph::opset8::FakeQuantize>(
                node,
                ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {-0.1}),
                ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {0.1}),
                ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {-0.1}),
                ngraph::opset8::Constant::create(ngraph::element::f32, {1}, {0.1}),
                255);
        }

        if (TRANSPOSE) {
            node = std::make_shared<ngraph::opset8::Transpose>(
                node,
                ngraph::opset8::Constant::create(ngraph::element::i64, {permutation_shape.size()}, permutation_shape));
        }

        return node;
    }

    static std::shared_ptr<ngraph::Function> CreateFunction(const ngraph::Shape& input_shape,
                                                            const ngraph::Shape& matmul_constant_shape,
                                                            const ngraph::Shape& permutation_shape = ngraph::Shape()) {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
        auto before = std::make_shared<ngraph::opset8::Relu>(input);
        auto matmul = CreateMatmul(before, matmul_constant_shape, permutation_shape);
        auto after = std::make_shared<ngraph::opset8::Relu>(matmul);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{std::make_shared<ngraph::opset8::Result>(after)},
                                                  ngraph::ParameterVector{input});
    }

    static std::shared_ptr<ngraph::Function> CreateReferenceFunction(
        const ngraph::Shape& input_shape,
        const std::vector<int>& reshape_before_shape,
        const ngraph::Shape& matmul_constant_shape,
        const ngraph::Shape& reshape_after_shape,
        const ngraph::Shape& permutation_shape = ngraph::Shape()) {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
        auto before = std::make_shared<ngraph::opset8::Relu>(input);
        auto reshape_before_constant = ngraph::opset8::Constant::create(ngraph::element::i64,
                                                                        ngraph::Shape{reshape_before_shape.size()},
                                                                        reshape_before_shape);
        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(before, reshape_before_constant, false);
        auto matmul = CreateMatmul(reshape_before, matmul_constant_shape, permutation_shape);
        auto reshape_after_constant = ngraph::opset8::Constant::create(ngraph::element::i64,
                                                                       ngraph::Shape{reshape_after_shape.size()},
                                                                       reshape_after_shape);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(matmul, reshape_after_constant, false);
        auto after = std::make_shared<ngraph::opset8::Relu>(reshape_after);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{std::make_shared<ngraph::opset8::Result>(after)},
                                                  ngraph::ParameterVector{input});
    }
};  // struct InsertReshapeAroundMatmulTest

namespace {

void RunTest(const std::shared_ptr<ngraph::Function>& func, const std::shared_ptr<ngraph::Function>& reference_func) {
    {
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::InsertReshapeAroundMatmulWithTranspose>();
        m.register_pass<ov::intel_gna::pass::InsertReshapeAroundMatmulWithFq>();
        m.register_pass<ov::intel_gna::pass::InsertReshapeAroundMatmulWithAdd>();
        m.register_pass<ov::intel_gna::pass::InsertReshapeAroundMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

}  // namespace

TEST(TransformationTests, InsertReshapeAroundMatmul) {
    RunTest(InsertReshapeAroundMatmulTest<>::CreateFunction({1, 6, 8}, {8, 10}),
            InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}));
    RunTest(InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}),
            InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}));
    RunTest(InsertReshapeAroundMatmulTest<>::CreateFunction({1, 6, 1, 8}, {8, 10}),
            InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 6, 1, 8}, {-1, 8}, {8, 10}, {1, 6, 1, 10}));
    RunTest(InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 6, 1, 8}, {-1, 8}, {8, 10}, {1, 6, 1, 10}),
            InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 6, 1, 8}, {-1, 8}, {8, 10}, {1, 6, 1, 10}));
    RunTest(InsertReshapeAroundMatmulTest<>::CreateFunction({1, 1, 8}, {8, 10}),
            InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 1, 8}, {-1, 8}, {8, 10}, {1, 1, 10}));
    RunTest(InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 1, 8}, {-1, 8}, {8, 10}, {1, 1, 10}),
            InsertReshapeAroundMatmulTest<>::CreateReferenceFunction({1, 1, 8}, {-1, 8}, {8, 10}, {1, 1, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithAdd) {
    RunTest(
        InsertReshapeAroundMatmulTest<true, true>::CreateFunction({1, 6, 8}, {8, 10}),
        InsertReshapeAroundMatmulTest<true, true>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<true, true, true>::CreateFunction({1, 6, 8}, {8, 10}),
        InsertReshapeAroundMatmulTest<true, true>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<true, true>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<true, true>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithAdd_AddFirstInputConstant) {
    RunTest(InsertReshapeAroundMatmulTest<true>::CreateFunction({1, 6, 8}, {8, 10}),
            InsertReshapeAroundMatmulTest<true>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}));
    RunTest(InsertReshapeAroundMatmulTest<true, false, true>::CreateFunction({1, 6, 8}, {8, 10}),
            InsertReshapeAroundMatmulTest<true>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}));
    RunTest(InsertReshapeAroundMatmulTest<true>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}),
            InsertReshapeAroundMatmulTest<true>::CreateReferenceFunction({1, 6, 8}, {-1, 8}, {8, 10}, {1, 6, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithFq) {
    RunTest(InsertReshapeAroundMatmulTest<false, false, false, true>::CreateFunction({1, 6, 8}, {8, 10}),
            InsertReshapeAroundMatmulTest<false, false, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                              {-1, 8},
                                                                                              {8, 10},
                                                                                              {1, 6, 10}));
    RunTest(InsertReshapeAroundMatmulTest<false, false, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                              {-1, 8},
                                                                                              {8, 10},
                                                                                              {1, 6, 10}),
            InsertReshapeAroundMatmulTest<false, false, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                              {-1, 8},
                                                                                              {8, 10},
                                                                                              {1, 6, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithAddAndFq) {
    RunTest(InsertReshapeAroundMatmulTest<true, true, false, true>::CreateFunction({1, 6, 8}, {8, 10}),
            InsertReshapeAroundMatmulTest<true, true, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                            {-1, 8},
                                                                                            {8, 10},
                                                                                            {1, 6, 10}));
    RunTest(InsertReshapeAroundMatmulTest<true, true, true, true>::CreateFunction({1, 6, 8}, {8, 10}),
            InsertReshapeAroundMatmulTest<true, true, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                            {-1, 8},
                                                                                            {8, 10},
                                                                                            {1, 6, 10}));
    RunTest(InsertReshapeAroundMatmulTest<true, true, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                            {-1, 8},
                                                                                            {8, 10},
                                                                                            {1, 6, 10}),
            InsertReshapeAroundMatmulTest<true, true, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                            {-1, 8},
                                                                                            {8, 10},
                                                                                            {1, 6, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithTranspose) {
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, false, false, true>::CreateFunction({1, 6, 8}, {8, 10}, {0, 2, 1}),
        InsertReshapeAroundMatmulTest<false, false, false, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                 {-1, 8},
                                                                                                 {8, 10},
                                                                                                 {1, 10, 6},
                                                                                                 {1, 0}));
    RunTest(InsertReshapeAroundMatmulTest<false, false, false, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                     {-1, 8},
                                                                                                     {8, 10},
                                                                                                     {1, 10, 6},
                                                                                                     {1, 0}),
            InsertReshapeAroundMatmulTest<false, false, false, false, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                     {-1, 8},
                                                                                                     {8, 10},
                                                                                                     {1, 10, 6},
                                                                                                     {1, 0}));
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, false, false, true>::CreateFunction({1, 1, 8}, {8, 10}, {0, 2, 1}),
        InsertReshapeAroundMatmulTest<false, false, false, false, true>::CreateReferenceFunction({1, 1, 8},
                                                                                                 {-1, 8},
                                                                                                 {8, 10},
                                                                                                 {1, 10, 1},
                                                                                                 {1, 0}));
    RunTest(InsertReshapeAroundMatmulTest<false, false, false, false, true>::CreateReferenceFunction({1, 1, 8},
                                                                                                     {-1, 8},
                                                                                                     {8, 10},
                                                                                                     {1, 10, 1},
                                                                                                     {1, 0}),
            InsertReshapeAroundMatmulTest<false, false, false, false, true>::CreateReferenceFunction({1, 1, 8},
                                                                                                     {-1, 8},
                                                                                                     {8, 10},
                                                                                                     {1, 10, 1},
                                                                                                     {1, 0}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithFqAndTranspose) {
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, false, true, true>::CreateFunction({1, 6, 8}, {8, 10}, {0, 2, 1}),
        InsertReshapeAroundMatmulTest<false, false, false, true, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                {-1, 8},
                                                                                                {8, 10},
                                                                                                {1, 10, 6},
                                                                                                {1, 0}));
    RunTest(InsertReshapeAroundMatmulTest<false, false, false, true, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                    {-1, 8},
                                                                                                    {8, 10},
                                                                                                    {1, 10, 6},
                                                                                                    {1, 0}),
            InsertReshapeAroundMatmulTest<false, false, false, true, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                    {-1, 8},
                                                                                                    {8, 10},
                                                                                                    {1, 10, 6},
                                                                                                    {1, 0}));
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, false, true, true>::CreateFunction({1, 1, 8}, {8, 10}, {0, 2, 1}),
        InsertReshapeAroundMatmulTest<false, false, false, true, true>::CreateReferenceFunction({1, 1, 8},
                                                                                                {-1, 8},
                                                                                                {8, 10},
                                                                                                {1, 10, 1},
                                                                                                {1, 0}));
    RunTest(InsertReshapeAroundMatmulTest<false, false, false, true, true>::CreateReferenceFunction({1, 1, 8},
                                                                                                    {-1, 8},
                                                                                                    {8, 10},
                                                                                                    {1, 10, 1},
                                                                                                    {1, 0}),
            InsertReshapeAroundMatmulTest<false, false, false, true, true>::CreateReferenceFunction({1, 1, 8},
                                                                                                    {-1, 8},
                                                                                                    {8, 10},
                                                                                                    {1, 10, 1},
                                                                                                    {1, 0}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithAddAndFqAndTranspose) {
    RunTest(InsertReshapeAroundMatmulTest<true, true, false, true, true>::CreateFunction({1, 6, 8}, {8, 10}, {0, 2, 1}),
            InsertReshapeAroundMatmulTest<true, true, false, true, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                  {-1, 8},
                                                                                                  {8, 10},
                                                                                                  {1, 10, 6},
                                                                                                  {1, 0}));
    RunTest(InsertReshapeAroundMatmulTest<true, true, true, true, true>::CreateFunction({1, 6, 8}, {8, 10}, {0, 2, 1}),
            InsertReshapeAroundMatmulTest<true, true, false, true, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                  {-1, 8},
                                                                                                  {8, 10},
                                                                                                  {1, 10, 6},
                                                                                                  {1, 0}));
    RunTest(InsertReshapeAroundMatmulTest<true, true, false, true, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                  {-1, 8},
                                                                                                  {8, 10},
                                                                                                  {1, 10, 6},
                                                                                                  {1, 0}),
            InsertReshapeAroundMatmulTest<true, true, false, true, true>::CreateReferenceFunction({1, 6, 8},
                                                                                                  {-1, 8},
                                                                                                  {8, 10},
                                                                                                  {1, 10, 6},
                                                                                                  {1, 0}));
}
