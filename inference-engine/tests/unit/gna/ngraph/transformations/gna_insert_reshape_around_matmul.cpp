// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/insert_reshape_around_matmul.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <numeric>

template<bool ADD, bool ADD_FIRST_INPUT_NOT_CONSTANT, bool FQ>
struct InsertReshapeAroundMatmulTest {
    static std::shared_ptr<ngraph::Node> CreateAdd(std::shared_ptr<ngraph::Node> input, const ngraph::Shape& constant_shape) {
        std::vector<size_t> data(ngraph::shape_size(constant_shape));
        std::iota(std::begin(data), std::end(data), 1);
        auto constant = ngraph::opset8::Constant::create(ngraph::element::i64, constant_shape, data);
        return std::make_shared<ngraph::opset8::Add>(input, constant);
    }

    static std::shared_ptr<ngraph::Node> CreateMatmul(
        std::shared_ptr<ngraph::Node> input,
        const ngraph::Shape& matmul_constant_shape) {
        std::vector<size_t> data(ngraph::shape_size(matmul_constant_shape));
        std::iota(std::begin(data), std::end(data), 1);
        auto constant = ngraph::opset8::Constant::create(ngraph::element::i64, matmul_constant_shape, data);
        std::shared_ptr<ngraph::Node> node;
        node = std::make_shared<ngraph::opset8::MatMul>(input, constant);

        if (ADD) {
            auto matmul_shape = node->get_output_shape(0);
            data.resize(ngraph::shape_size(matmul_shape));
            std::iota(std::begin(data), std::end(data), 1);
            std::vector<size_t> constant_add_shape(2, 1);
            std::copy_if(matmul_shape.begin(), matmul_shape.end(), constant_add_shape.begin(), [](size_t e) { return e > 1; });
            auto constant_add = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{constant_add_shape}, data);
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

        return node;
    }

    static std::shared_ptr<ngraph::Function> CreateFunction(
        const ngraph::Shape& input_shape,
        const ngraph::Shape& matmul_constant_shape,
        const ngraph::Shape& result_shape) {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
        auto before = std::make_shared<ngraph::opset8::Relu>(input);
        auto matmul = CreateMatmul(before, matmul_constant_shape);
        auto after = std::make_shared<ngraph::opset8::Relu>(matmul);
        return std::make_shared<ngraph::Function>(
            ngraph::ResultVector{std::make_shared<ngraph::opset8::Result>(after)},
            ngraph::ParameterVector{input});
    }

    static std::shared_ptr<ngraph::Function> CreateReferenceFunction(
        const ngraph::Shape& input_shape,
        const ngraph::Shape& reshape_before_shape,
        const ngraph::Shape& matmul_constant_shape,
        const ngraph::Shape& reshape_after_shape,
        const ngraph::Shape& result_shape) {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
        auto before = std::make_shared<ngraph::opset8::Relu>(input);
        auto reshape_before_constant = ngraph::opset8::Constant::create(ngraph::element::i64,
            ngraph::Shape{reshape_before_shape.size()}, reshape_before_shape);
        auto reshape_before = std::make_shared<ngraph::opset8::Reshape>(before, reshape_before_constant, false);
        auto matmul = CreateMatmul(reshape_before, matmul_constant_shape);
        auto reshape_after_constant = ngraph::opset8::Constant::create(ngraph::element::i64,
            ngraph::Shape{reshape_after_shape.size()}, reshape_after_shape);
        auto reshape_after = std::make_shared<ngraph::opset8::Reshape>(matmul, reshape_after_constant, false);
        auto after = std::make_shared<ngraph::opset8::Relu>(reshape_after);
        return std::make_shared<ngraph::Function>(
            ngraph::ResultVector{std::make_shared<ngraph::opset8::Result>(after)},
            ngraph::ParameterVector{input});
    }
}; // struct InsertReshapeAroundMatmulTest

namespace {

void RunTest(const std::shared_ptr<ngraph::Function>& func, const std::shared_ptr<ngraph::Function>& reference_func) {
    {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertReshapeAroundMatmulWithTranspose>();
        m.register_pass<GNAPluginNS::InsertReshapeAroundMatmulWithFq>();
        m.register_pass<GNAPluginNS::InsertReshapeAroundMatmulWithAdd>();
        m.register_pass<GNAPluginNS::InsertReshapeAroundMatmul>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace

TEST(TransformationTests, InsertReshapeAroundMatmul) {
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, false>::
            CreateFunction({1, 6, 8}, {8, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<false, false, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<false, false, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, false>::
            CreateFunction({1, 6, 1, 8}, {8, 10}, {1, 6, 1, 10}),
        InsertReshapeAroundMatmulTest<false, false, false>::
            CreateReferenceFunction({1, 6, 1, 8}, {6, 8}, {8, 10}, {1, 6, 1, 10}, {1, 6, 1, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, false>::
            CreateReferenceFunction({1, 6, 1, 8}, {6, 8}, {8, 10}, {1, 6, 1, 10}, {1, 6, 1, 10}),
        InsertReshapeAroundMatmulTest<false, false, false>::
            CreateReferenceFunction({1, 6, 1, 8}, {6, 8}, {8, 10}, {1, 6, 1, 10}, {1, 6, 1, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithAdd) {
    RunTest(
        InsertReshapeAroundMatmulTest<true, true, false>::
            CreateFunction({1, 6, 8}, {8, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<true, true, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<true, true, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<true, true, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithAdd_AddFirstInputConstant) {
    RunTest(
        InsertReshapeAroundMatmulTest<true, false, false>::
            CreateFunction({1, 6, 8}, {8, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<true, false, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<true, false, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<true, false, false>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithFq) {
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, true>::
            CreateFunction({1, 6, 8}, {8, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<false, false, true>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<false, false, true>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<false, false, true>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
}

TEST(TransformationTests, InsertReshapeAroundMatmulWithAddAndFq) {
    RunTest(
        InsertReshapeAroundMatmulTest<true, true, true>::
            CreateFunction({1, 6, 8}, {8, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<true, true, true>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
    RunTest(
        InsertReshapeAroundMatmulTest<true, true, true>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}),
        InsertReshapeAroundMatmulTest<true, true, true>::
            CreateReferenceFunction({1, 6, 8}, {6, 8}, {8, 10}, {1, 6, 10}, {1, 6, 10}));
}
