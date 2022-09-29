// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph_transformations/op/fully_connected.hpp>
#include <ngraph_transformations/convert_matmul_to_fc.hpp>
#include <ngraph_transformations/fc_bias_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST(TransformationTests, ConvertMatMulToFCTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 });
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 2, 2 }, { 1 });
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, true, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 });
        auto transpose_constant = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 3 }, { 0, 2, 1 });
        auto transpose = std::make_shared<ngraph::opset1::Transpose>(input1, transpose_constant);
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 2 }, { 1 });
        auto matmul = std::make_shared<FullyConnectedNode>(transpose, input2, ngraph::Rank(3));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(3));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(3));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest5) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{ -1, -1, 2 });
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 }, { 1 });
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ConvertMatMulToFC>();
    ASSERT_NO_THROW(m.run_passes(f));
}

TEST(TransformationTests, ConvertMatMulToFCTest6) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{ -1, -1, 2 });
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 2 }, { 1 });
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ConvertMatMulToFC>();
    ASSERT_NO_THROW(m.run_passes(f));
    ASSERT_NO_THROW(check_rt_info(f));
}

TEST(TransformationTests, ConvertMatMulToFCTest7) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(2));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest8) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});

        auto fc = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(2));
        auto a_shape = std::make_shared<ngraph::opset3::ShapeOf>(input1);

        auto I = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_node(a_shape, {0, 1});
        auto O = ngraph::opset1::Constant::create(ngraph::element::i64, { 1 }, { 3 });
        auto output_shape = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{I, O}, 0);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest9) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});

        ngraph::pass::Manager m;
        auto pass_config = m.get_pass_config();
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(3));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest10) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 2 }, { 1 });
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ConvertMatMulToFC>();
    ASSERT_NO_THROW(m.run_passes(f));
}

TEST(TransformationTests, FullyConnectedBiasFusionTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(3));

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        ASSERT_NO_THROW(manager.run_passes(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(3));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FullyConnectedBiasFusionTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(3));

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        ASSERT_NO_THROW(manager.run_passes(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(3));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FullyConnectedBiasFusionTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2));

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        ASSERT_NO_THROW(manager.run_passes(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(2));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FullyConnectedBiasFusionTest4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2));

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        ASSERT_NO_THROW(manager.run_passes(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(2));

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FullyConnectedBiasFusionTest5) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 786, 128 }, { 1 });
    auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2));

    auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 786 }, { 1 });
    auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input1 });

    ngraph::pass::Manager manager;
    manager.register_pass<FullyConnectedBiasFusion>();
    ASSERT_NO_THROW(manager.run_passes(f));
}

TEST(TransformationTests, FullyConnectedBiasFusionTest6) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::PartialShape{ -1, -1 });
    auto weights = ngraph::opset1::Constant::create(ngraph::element::i8, ngraph::Shape{ 786, 128 }, { 1 });
    auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2), ngraph::element::f32);

    auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 786 }, { 1 });
    auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input1 });

    ngraph::pass::Manager manager;
    manager.register_pass<FullyConnectedBiasFusion>();
    ASSERT_NO_THROW(manager.run_passes(f));
}

TEST(TransformationTests, ConvertMatMulToFCTest_second_input_rank_adj_1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 2, 3});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 2, 3}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 2, 3});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 3}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(2));
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest_second_input_rank_adj_2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 2, 3 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 3 }, { 1 });
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, weights, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 2, 3 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 3 }, { 1 });
        auto matmul = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2));
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulToFCTest_second_input_rank_adj_3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 5, 2, 3 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 2, 3 }, { 1 });
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, weights, false, true);
        auto biases = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 1, 2 }, { 1 });
        auto add = std::make_shared<ngraph::opset1::Add>(matmul, biases);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input1 });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<FullyConnectedBiasFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 5, 2, 3 });
        auto reshape_before_const = ngraph::opset1::Constant::create(ngraph::element::i64, { 2 }, { -1, 3 });

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 3 }, { 1 });
        auto biases = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2 }, { 1 });
        auto matmul = std::make_shared<FullyConnectedNode>(input1, weights, biases, ngraph::Rank(2));

        auto reshape_after_const = ngraph::opset1::Constant::create(ngraph::element::i64, { 4 }, { 1, 5, 2, 2 });
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
