// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_transformations/op/fully_connected.hpp>
#include <ngraph_transformations/convert_matmul_to_fc_or_gemm.hpp>
#include <ngraph_transformations/fc_bias_fusion.hpp>
#include <ngraph_transformations/reshape_fully_connected.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace MKLDNNPlugin;

TEST(TransformationTests, ConvertMatMulTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});

        auto reshape = ngraph::op::util::reshapeTo(input2, {1, 2, 1});

        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, reshape, false, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});

        auto usnqueeze_input2 = std::make_shared<ngraph::opset1::Unsqueeze>(input2,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}));
        auto reshape = ngraph::op::util::reshapeTo(usnqueeze_input2, {1, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, reshape, false, false);
        auto reshape_output = ngraph::op::util::reshapeTo(matmul, {3, 1});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_output}, ngraph::ParameterVector{input1, input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});

        auto usnqueeze_input1 = std::make_shared<ngraph::opset1::Unsqueeze>(input1,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto reshape = ngraph::op::util::reshapeTo(usnqueeze_input1, {1, 1, 2});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(reshape, input2, false, false);
        auto reshape_output = ngraph::op::util::reshapeTo(matmul, {3, 1});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_output}, ngraph::ParameterVector{input1, input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
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

TEST(TransformationTests, ConvertMatMulTest5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Shape{3, 2, 2});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest5_dynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::PartialShape{-1, -1, 2});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest6) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.register_pass<ReshapeFullyConnected>();
        m.register_pass<ngraph::pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto reshape_begin = std::make_shared<ngraph::opset1::Reshape>(
                input1, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<int64_t>{-1, 2}), false);
        auto fc = std::make_shared<FullyConnectedNode>(reshape_begin, input2, ngraph::Shape{6, 3});
        auto reshape_end = ngraph::op::util::reshapeTo(fc, ngraph::Shape{3, 2, 3});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_end}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest6_dynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.register_pass<ReshapeFullyConnected>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});

        auto reshape_begin = std::make_shared<ngraph::opset1::Reshape>(
                input1, ngraph::opset1::Constant::create(ngraph::element::i64, {2}, {-1, 2}), false);

        auto fc = std::make_shared<FullyConnectedNode>(reshape_begin, input2, ngraph::PartialShape{-1, 3});
        auto a_shape = std::make_shared<ngraph::opset3::ShapeOf>(input1);

        auto I = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_node(a_shape, {0, 1});
        auto b_shape = std::make_shared<ngraph::opset3::ShapeOf>(input2);
        auto O = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_node(b_shape, {0});
        auto output_shape = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{I, O}, 0);
        auto reshape_end = std::make_shared<ngraph::opset1::Reshape>(fc, output_shape, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_end}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest7) {
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
        m.register_pass<ConvertMatMulToGemm>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Shape{3, 2, 2});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulDynamic) {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.register_pass<ReshapeFullyConnected>();
        ASSERT_NO_THROW(m.run_passes(f));
}


TEST(TransformationTests, FullyConnectedBiasFusionTest3D) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Shape{1, 128, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Shape{1, 128, 786});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FullyConnectedBiasFusionTest3D_dynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::PartialShape{-1, -1, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::PartialShape{-1, -1, 786});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FullyConnectedBiasFusionTest2D) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Shape{1, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Shape{1, 786});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}


TEST(TransformationTests, FullyConnectedBiasFusionTest2D_dynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::PartialShape{-1, 786});

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<FullyConnectedBiasFusion>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::PartialShape{-1, 786});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FullyConnectedBiasFusionDynamic) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
    auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Shape{1, 786});

    auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
    auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
    ngraph::pass::Manager manager;
    manager.register_pass<FullyConnectedBiasFusion>();
    ASSERT_NO_THROW(manager.run_passes(f));
}

TEST(TransformationTests, ConvertMatMulTest_second_input_rank_adj) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 2, 3});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 2, 3}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.register_pass<ReshapeFullyConnected>();
        m.register_pass<ngraph::pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 2, 3});
        auto reshape_1 = std::make_shared<ngraph::opset1::Reshape>(input1, ngraph::opset1::Constant::create(ngraph::element::i64, {2}, {-1, 3}), false);
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 3}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(reshape_1, input2, ngraph::Shape{10, 2});
        auto reshape_out = std::make_shared<ngraph::opset1::Reshape>(matmul, ngraph::opset1::Constant::create(ngraph::element::i64, {4}, {1, 5, 2, 2}), false);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_out}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest_second_input_rank_adj_dynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, 2, 3});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 2, 3}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertMatMulToFC>();
        m.register_pass<ConvertMatMulToGemm>();
        m.register_pass<ReshapeFullyConnected>();
        m.register_pass<ngraph::pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, 2, 3});
        auto reshape_1 = std::make_shared<ngraph::opset1::Reshape>(input1, ngraph::opset1::Constant::create(ngraph::element::i64, {2}, {-1, 3}), false);
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 3}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(reshape_1, input2, ngraph::PartialShape{-1, 2});

        auto shape_of = std::make_shared<ngraph::opset7::ShapeOf>(input1);
        auto gather = std::make_shared<ngraph::opset7::Gather>(
                shape_of, ngraph::opset1::Constant::create(ngraph::element::i64, {2}, {0, 1}), ngraph::opset1::Constant::create(ngraph::element::i64, {}, {0}));
        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
                ngraph::opset1::Constant::create(ngraph::element::i64, {1}, {1}),
                gather,
                ngraph::opset1::Constant::create(ngraph::element::i64, {1}, {2}),
        }, 0);
        auto reshape_out = std::make_shared<ngraph::opset1::Reshape>(matmul, concat, false);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_out}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
