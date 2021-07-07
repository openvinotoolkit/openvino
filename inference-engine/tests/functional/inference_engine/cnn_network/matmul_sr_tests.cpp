// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <map>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/function.hpp>
#include <common_test_utils/ngraph_test_utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/smart_reshape/matmul_sr.hpp>

#include "cnn_network_ngraph_impl.hpp"

using namespace testing;
using namespace InferenceEngine;

namespace {

using reshape_map = std::map<std::string, std::vector<size_t>>;

struct ReshapeMatMulTestCase {
    bool reshape_is_A_input;
    ngraph::PartialShape A_shape, B_shape;
    std::vector<int64_t> reshape_pattern;
    bool transpose_a, transpose_b;
    reshape_map new_shapes;
};

class SmartReshapeMatMulTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<ReshapeMatMulTestCase>> {
public:
    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());

        std::shared_ptr<ngraph::Function> ngraph;
        {
            auto input_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.A_shape);
            input_A->set_friendly_name("input_A");
            auto input_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.B_shape);
            input_B->set_friendly_name("input_B");

            auto reshape_pattern = std::make_shared<ngraph::opset4::Constant>(
                    ngraph::element::i64, ngraph::Shape{test_case.reshape_pattern.size()}, test_case.reshape_pattern);
            reshape_pattern->set_friendly_name("reshape_pattern");
            auto reshape = std::make_shared<ngraph::opset4::Reshape>(test_case.reshape_is_A_input ? input_A : input_B, reshape_pattern, true);
            reshape->set_friendly_name("reshape");

            auto mat_mul = std::make_shared<ngraph::opset4::MatMul>(test_case.reshape_is_A_input ? reshape->output(0) : input_A->output(0),
                                                                    test_case.reshape_is_A_input ? input_B->output(0) : reshape->output(0),
                                                                    test_case.transpose_a, test_case.transpose_b);
            reshape->set_friendly_name("matmul");

            auto result = std::make_shared<ngraph::op::Result>(mat_mul);
            ngraph::ParameterVector params = {input_A, input_B};
            ngraph::ResultVector results = {result};
            ngraph = std::make_shared<ngraph::Function>(results, params);
        }

        InferenceEngine::details::CNNNetworkNGraphImpl network(ngraph);
        const auto & resp = network.reshape(test_case.new_shapes, nullptr);
        ASSERT_EQ(resp, StatusCode::OK);
    }
};

TEST_P(SmartReshapeMatMulTests, ReshapeMatMul) {
}

INSTANTIATE_TEST_SUITE_P(NGraph, SmartReshapeMatMulTests, testing::Values(
        ReshapeMatMulTestCase{true, {1, 20, 30}, {30, 40}, {20, -1}, false, false, {{"input_A", {2, 20, 30}}}},
        ReshapeMatMulTestCase{true, {1, 20, 30}, {40, 30}, {20, -1}, false, true, {{"input_A", {2, 20, 30}}}},
        ReshapeMatMulTestCase{true, {1, 30, 20}, {30, 20}, {-1, 20}, true, false, {{"input_A", {2, 30, 20}}}},
        ReshapeMatMulTestCase{true, {1, 30, 20}, {40, 30}, {-1, 20}, true, true, {{"input_A", {2, 30, 20}}}},
        ReshapeMatMulTestCase{false, {20, 30}, {1, 30, 40}, {-1, 40}, false, false, {{"input_B", {2, 30, 40}}}},
        ReshapeMatMulTestCase{false, {20, 30}, {1, 40, 30}, {40, -1}, false, true, {{"input_B", {2, 40, 30}}}},
        ReshapeMatMulTestCase{false, {30, 20}, {1, 30, 40}, {-1, 40}, true, false, {{"input_B", {2, 30, 40}}}},
        ReshapeMatMulTestCase{false, {30, 20}, {1, 40, 30}, {40, -1}, true, true, {{"input_B", {2, 40, 30}}}}));
}  // namespace

TEST(SmartReshapeTransposeMatMulTests, TransposeAMatMulFuse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 2});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto order = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ngraph::opset4::Transpose>(data_A, order);
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(transpose, data_B, false, false);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeMatMul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 2});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, data_B, true, false);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeBMatMulFuse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 5, 3});
        auto order = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ngraph::opset4::Transpose>(data_B, order);
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, transpose, false, false);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeMatMul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 5, 3});
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, data_B, false, true);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeAMatMulWithAttrFuse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto order = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ngraph::opset4::Transpose>(data_A, order);
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(transpose, data_B, true, false);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeMatMul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, data_B, false, false);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeBMatMulWithAttrFuse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto order = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ngraph::opset4::Transpose>(data_B, order);
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, transpose, false, true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeMatMul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, data_B, false, false);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
TEST(SmartReshapeTransposeMatMulTests, TransposeAMatMulSideAttrFuse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 5, 3});
        auto order = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ngraph::opset4::Transpose>(data_A, order);
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(transpose, data_B, true, true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeMatMul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 5, 3});
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, data_B, false, true);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeBMatMulSideAttrFuse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 2});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto order = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ngraph::opset4::Transpose>(data_B, order);
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, transpose, true, true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeMatMul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 2});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, data_B, true, false);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeBothMatMulFuse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 2});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 5, 3});
        auto order = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});
        auto transpose_A = std::make_shared<ngraph::opset4::Transpose>(data_A, order);
        auto transpose_B = std::make_shared<ngraph::opset4::Transpose>(data_B, order);
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(transpose_A, transpose_B, false, false);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeMatMul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 2});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 5, 3});
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, data_B, true, true);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
TEST(SmartReshapeTransposeMatMulTests, TransposeBothMatMulWithAttrFuse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 2});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto order = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});
        auto transpose_A = std::make_shared<ngraph::opset4::Transpose>(data_A, order);
        auto transpose_B = std::make_shared<ngraph::opset4::Transpose>(data_B, order);
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(transpose_A, transpose_B, false, true);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeMatMul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 2});
        auto data_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5});
        auto matmul = std::make_shared<ngraph::opset4::MatMul>(data_A, data_B, true, false);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
