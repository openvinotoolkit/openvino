// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph_transformations/move_eltwise_up_data_movement.hpp>

using namespace testing;

TEST(MoveEltwiseUpThroughDataMov, SingleUnaryEltwise) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid}, ngraph::ParameterVector{input});
    }

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    std::shared_ptr<ngraph::Function> f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(input);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMov, EltwiseSequence) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {1, 2, 0, 3};
    const int64_t unsqueeze_axis = 1;
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input_left = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);
        auto input_right = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto matmul = std::make_shared<ngraph::opset8::MatMul>(input_left, input_right);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(matmul, transpose_const);

        auto relu = std::make_shared<ngraph::opset8::Relu>(transpose);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(relu, unsqueeze_const);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid}, ngraph::ParameterVector{input_left, input_right});
    }

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    std::shared_ptr<ngraph::Function> f_ref(nullptr);
    {
        auto input_left = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);
        auto input_right = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto matmul = std::make_shared<ngraph::opset8::MatMul>(input_left, input_right);

        auto relu = std::make_shared<ngraph::opset8::Relu>(matmul);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(relu);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input_left, input_right});
    }

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMov, DataMovementTwoConsumers) {
    /* In this case transformation shouldn't apply */
    auto create_graph = [] () -> std::shared_ptr<ngraph::Function> {
        const ngraph::Shape shape{1, 3, 224, 224};
        const std::vector<int64_t> input_order = {1, 2, 0, 3};
        const int64_t unsqueeze_axis = 1;

        auto input_left = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);
        auto input_right = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto matmul = std::make_shared<ngraph::opset8::MatMul>(input_left, input_right);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(matmul, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);

        auto relu = std::make_shared<ngraph::opset8::Relu>(transpose);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid, relu}, ngraph::ParameterVector{input_left, input_right});
    };

    std::shared_ptr<ngraph::Function> f = create_graph();

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    std::shared_ptr<ngraph::Function> f_ref = create_graph();

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMov, SingleBinaryEltwiseWithScalarOnSecondBranch) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto add = std::make_shared<ngraph::opset8::Add>(unsqueeze, ngraph::opset8::Constant::create(ngraph::element::f32, {}, {scalar_value}));

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});
    }
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    std::shared_ptr<ngraph::Function> f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto add = std::make_shared<ngraph::opset8::Add>(input, ngraph::opset8::Constant::create(ngraph::element::f32, {}, {scalar_value}));

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(add, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMov, SingleEltwiseWith5ScalarOnSecondBranch) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(input, unsqueeze_const);

        auto add = std::make_shared<ngraph::opset8::Add>(unsqueeze, ngraph::opset8::Constant::create(ngraph::element::f32, {1, 1, 1, 1, 1}, {scalar_value}));

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});
    }
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    std::shared_ptr<ngraph::Function> f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto add = std::make_shared<ngraph::opset8::Add>(input, ngraph::opset8::Constant::create(ngraph::element::f32, {}, {scalar_value}));

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(add, unsqueeze_const);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMov, SingleBinaryEltwiseWithNotScalarOnSecondBranch) {
    auto create_graph = [] () -> std::shared_ptr<ngraph::Function> {
        const ngraph::Shape shape{1, 3, 224, 224};
        const std::vector<int64_t> input_order = {3, 2, 1, 0};
        const int64_t unsqueeze_axis = 2;
        std::shared_ptr<ngraph::Function> f(nullptr);
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto add_scalar = ngraph::opset8::Constant::create(ngraph::element::f32, {1, 1, 1, 3}, {0.5, 0.2, 0.3});
        auto add = std::make_shared<ngraph::opset8::Add>(unsqueeze, add_scalar);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});
    };
    std::shared_ptr<ngraph::Function> f = create_graph();
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    std::shared_ptr<ngraph::Function> f_ref = create_graph();
    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMov, SingleUnaryEltwiseDynamicShape) {
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(input, unsqueeze_const);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid}, ngraph::ParameterVector{input});
    }

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    std::shared_ptr<ngraph::Function> f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(input);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(sigmoid, unsqueeze_const);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMov, SingleUnaryEltwiseDynamicRank) {
    auto create_graph = [] () -> std::shared_ptr<ngraph::Function> {
        const std::vector<int64_t> input_order = {3, 2, 1, 0};
        const int64_t unsqueeze_axis = 2;
        std::shared_ptr<ngraph::Function> f(nullptr);
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(ngraph::Rank::dynamic()));

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(input, unsqueeze_const);
        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid}, ngraph::ParameterVector{input});
    };
    std::shared_ptr<ngraph::Function> f = create_graph();
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();

    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    std::shared_ptr<ngraph::Function> f_ref = create_graph();
    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}
