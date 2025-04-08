// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/fp16_compression/mark_subgraphs_to_keep_in_mixed_precision.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace testing;
using namespace ov;
using namespace std;
using namespace ov::opset10;

TEST(TransformationTests, keep_precission_sensitive_fp32_1) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp_1);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(factor_const_decompressed);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, keep_precission_sensitive_fp32_with_reducemean) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_mean_1 = make_shared<ReduceMean>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_mean_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp_1);
        disable_fp16_compression(reduce_mean_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(factor_const_decompressed);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkSugraphsToKeepInMixedPrecision_reducesum_without_exp) {
    // ReduceSum without Exp is not a precision sensitive case,
    // no nodes should be marked with disable_fp16_compression
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
    auto reduce_sum_1 = make_shared<ReduceSum>(input_1, reduction_axes);

    auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
    auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
    auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
    auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

    model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, keep_precission_sensitive_fp32_2) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp_1);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(unsqueeze_1);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(factor_const_decompressed);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, keep_precission_sensitive_fp32_3) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});

        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);
        auto addition_const = Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<Add>(reduce_sum_1, addition_const);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(add_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});

        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);
        auto addition_const = Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<Add>(reduce_sum_1, addition_const);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(add_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp_1);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(add_1);
        disable_fp16_compression(addition_const);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(factor_const_decompressed);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    //    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, keep_precission_sensitive_fp32_7) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // subgraph from t2t-vit-7
    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_3 = make_shared<Parameter>(element::f32, Shape{1, 3136, 64, 1});
        auto input_4 = make_shared<Parameter>(element::f32, Shape{128, 64});
        auto exp_1 = make_shared<Exp>(input_1);
        auto exp_2 = make_shared<Exp>(input_2);

        auto factor_1 = Constant::create(element::f32, Shape{1}, {0.5});  // add decompression
        auto mul_1 = make_shared<Multiply>(exp_1, factor_1);
        auto factor_2 = Constant::create(element::f32, Shape{1}, {0.5});
        auto mul_2 = make_shared<Multiply>(exp_2, factor_2);

        auto const_unsqueeze_1 = Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_1 = make_shared<Reshape>(mul_1, const_unsqueeze_1, false);

        auto const_unsqueeze_2 = Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_2 = make_shared<Reshape>(mul_2, const_unsqueeze_1, false);
        auto reduction_axes_1 = Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_1 = make_shared<ReduceSum>(mul_2, reduction_axes_1, true);
        auto mul_3 = make_shared<Multiply>(reduce_sum_1, mul_1);
        auto mul_4 = make_shared<Multiply>(input_3, unsqueeze_2);

        auto reduction_axes_2 = Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_2 = make_shared<ReduceSum>(mul_4, reduction_axes_2);
        auto reduction_axes_3 = Constant::create(element::i64, Shape{1}, {2});
        auto reduce_sum_3 = make_shared<ReduceSum>(mul_3, reduction_axes_3, true);

        auto broadcast_to_shape = Constant::create(element::i64, Shape{3}, {1, 1, 1});
        auto broadcast = make_shared<Broadcast>(reduce_sum_3, broadcast_to_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto tile_shape = Constant::create(element::i64, Shape{3}, {1, 1, 64});
        auto tile = make_shared<Tile>(broadcast, tile_shape);
        auto eps_const = Constant::create(element::f32, Shape{1}, {1.e-10});
        auto add_1 = make_shared<Add>(tile, eps_const);

        auto const_unsqueeze_3 = Constant::create(element::i64, Shape{4}, {1, 1, 64, 32});
        auto unsqueeze_3 = make_shared<Reshape>(reduce_sum_2, const_unsqueeze_3, false);
        auto mul_5 = make_shared<Multiply>(unsqueeze_1, unsqueeze_3);

        auto reduction_axes_4 = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_4 = make_shared<ReduceSum>(mul_5, reduction_axes_4);

        auto div_1 = make_shared<Divide>(reduce_sum_4, add_1);
        auto matmul_1 = make_shared<MatMul>(div_1, input_4, false, true);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2, input_3, input_4});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_3 = make_shared<Parameter>(element::f32, Shape{1, 3136, 64, 1});
        auto input_4 = make_shared<Parameter>(element::f32, Shape{128, 64});
        auto exp_1 = make_shared<Exp>(input_1);
        auto exp_2 = make_shared<Exp>(input_2);

        auto factor_1 = Constant::create(element::f32, Shape{1}, {0.5});  // add decompression
        auto mul_1 = make_shared<Multiply>(exp_1, factor_1);
        auto factor_2 = Constant::create(element::f32, Shape{1}, {0.5});
        auto mul_2 = make_shared<Multiply>(exp_2, factor_2);

        auto const_unsqueeze_1 = Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_1 = make_shared<Reshape>(mul_1, const_unsqueeze_1, false);

        auto const_unsqueeze_2 = Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_2 = make_shared<Reshape>(mul_2, const_unsqueeze_2, false);
        auto reduction_axes_1 = Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_1 = make_shared<ReduceSum>(mul_2, reduction_axes_1, true);
        auto mul_3 = make_shared<Multiply>(reduce_sum_1, mul_1);
        auto mul_4 = make_shared<Multiply>(input_3, unsqueeze_2);

        auto reduction_axes_2 = Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_2 = make_shared<ReduceSum>(mul_4, reduction_axes_2);
        auto reduction_axes_3 = Constant::create(element::i64, Shape{1}, {2});
        auto reduce_sum_3 = make_shared<ReduceSum>(mul_3, reduction_axes_3, true);

        auto broadcast_to_shape = Constant::create(element::i64, Shape{3}, {1, 1, 1});
        auto broadcast = make_shared<Broadcast>(reduce_sum_3, broadcast_to_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto tile_shape = Constant::create(element::i64, Shape{3}, {1, 1, 64});
        auto tile = make_shared<Tile>(broadcast, tile_shape);
        auto eps_const = Constant::create(element::f32, Shape{1}, {1.e-10});
        auto add_1 = make_shared<Add>(tile, eps_const);

        auto const_unsqueeze_3 = Constant::create(element::i64, Shape{4}, {1, 1, 64, 32});
        auto unsqueeze_3 = make_shared<Reshape>(reduce_sum_2, const_unsqueeze_3, false);
        auto mul_5 = make_shared<Multiply>(unsqueeze_1, unsqueeze_3);

        auto reduction_axes_4 = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_4 = make_shared<ReduceSum>(mul_5, reduction_axes_4);

        auto div_1 = make_shared<Divide>(reduce_sum_4, add_1);
        auto matmul_1 = make_shared<MatMul>(div_1, input_4, false, true);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(mul_1);
        disable_fp16_compression(mul_2);
        disable_fp16_compression(mul_3);
        disable_fp16_compression(mul_4);
        disable_fp16_compression(mul_5);
        disable_fp16_compression(unsqueeze_1);
        disable_fp16_compression(unsqueeze_2);
        disable_fp16_compression(unsqueeze_3);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(reduce_sum_2);
        disable_fp16_compression(reduce_sum_3);
        disable_fp16_compression(reduce_sum_4);
        disable_fp16_compression(exp_1);
        disable_fp16_compression(exp_2);
        disable_fp16_compression(tile);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(add_1);
        disable_fp16_compression(broadcast);
        disable_fp16_compression(div_1);

        disable_fp16_compression(factor_1);
        disable_fp16_compression(factor_2);

        disable_fp16_compression(broadcast_to_shape);
        disable_fp16_compression(tile_shape);
        disable_fp16_compression(const_unsqueeze_1);
        disable_fp16_compression(const_unsqueeze_2);
        disable_fp16_compression(const_unsqueeze_3);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2, input_3, input_4});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, DivisionByZeroMinimalPattern) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    const float eps_value = 1.0e-12f;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);
        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);
        disable_fp16_compression(divide);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(add);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, DivisionByZeroEpsWithConvert) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    const float eps_value = 1.0e-5f;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f16, Shape{1}, {eps_value});
        auto convert_eps = std::make_shared<Convert>(eps_const, element::f32);

        auto add = std::make_shared<Add>(input_2, convert_eps);
        auto divide = std::make_shared<Divide>(input_1, add);
        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f16, Shape{1}, {eps_value});
        auto convert_eps = std::make_shared<Convert>(eps_const, element::f32);
        auto add = std::make_shared<Add>(input_2, convert_eps);
        auto divide = std::make_shared<Divide>(input_1, add);
        disable_fp16_compression(divide);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(convert_eps);
        disable_fp16_compression(add);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, PowWithNegativeExponent) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    const float eps_value = 1.0e-12f;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto pow_exp_const = Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<Power>(add, pow_exp_const);
        auto mul = std::make_shared<Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto pow_exp_const = Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<Power>(add, pow_exp_const);
        auto mul = std::make_shared<Multiply>(input_1, pow);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(eps_const);
        disable_fp16_compression(add);
        disable_fp16_compression(pow_exp_const);
        disable_fp16_compression(pow);
        disable_fp16_compression(mul);

        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, PowWithPositiveExponent) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // graph should be left unchanged
    const float eps_value = 1.0e-12f;
    auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
    auto add = std::make_shared<Add>(input_2, eps_const);
    auto pow_exp_const = Constant::create(element::f32, Shape{1}, {1.77});
    auto pow = std::make_shared<Power>(add, pow_exp_const);
    auto mul = std::make_shared<Multiply>(input_1, pow);

    model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, DivisionByZeroMinimalPatternUnchanged) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // if eps_value is greater than normalized_fp16_min then leave graph unchanged
    const float eps_value = 0.0001f;
    auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
    auto add = std::make_shared<Add>(input_2, eps_const);
    auto divide = std::make_shared<Divide>(input_1, add);

    model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, DivisionByZeroInL2NormWithSqrtAndWithMax) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    const float eps_value = 1.0e-12f;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(max);
        auto divide = std::make_shared<Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(max);
        auto divide = std::make_shared<Divide>(input, sqrt);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp);
        disable_fp16_compression(pow);
        disable_fp16_compression(reduce_sum);
        disable_fp16_compression(max);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(sqrt);
        disable_fp16_compression(divide);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, DivisionByZeroMaxAndEpsWithConvert) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    const float eps_value = 1.0e-5f;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f16, Shape{}, {eps_value});
        auto convert_eps = std::make_shared<Convert>(eps_const, element::f32);
        auto max = std::make_shared<Maximum>(reduce_sum, convert_eps);
        auto sqrt = std::make_shared<Sqrt>(max);
        auto divide = std::make_shared<Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f16, Shape{}, {eps_value});
        auto convert_eps = std::make_shared<Convert>(eps_const, element::f32);
        auto max = std::make_shared<Maximum>(reduce_sum, convert_eps);
        auto sqrt = std::make_shared<Sqrt>(max);
        auto divide = std::make_shared<Divide>(input, sqrt);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp);
        disable_fp16_compression(pow);
        disable_fp16_compression(reduce_sum);
        disable_fp16_compression(max);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(convert_eps);
        disable_fp16_compression(sqrt);
        disable_fp16_compression(divide);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, DivisionByZeroInL2NormWithSqrtAndWithAdd) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    const float eps_value = 1.e-12f;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(add);
        auto divide = std::make_shared<Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(add);
        auto divide = std::make_shared<Divide>(input, sqrt);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp);
        disable_fp16_compression(pow);
        disable_fp16_compression(sqrt);
        disable_fp16_compression(reduce_sum);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(add);
        disable_fp16_compression(divide);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkReduceOpExpToKeepInMixedPrecision_with_reducesum) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
        disable_fp16_compression(exp_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(factor_const_decompressed);
        disable_fp16_compression(factor_const);
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkReduceOpExpToKeepInMixedPrecision_with_reducemean) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_mean_1 = make_shared<ReduceMean>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_mean_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_mean_1 = make_shared<ReduceMean>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_mean_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
        disable_fp16_compression(exp_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(reduce_mean_1);
        disable_fp16_compression(factor_const_decompressed);
        disable_fp16_compression(factor_const);
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkReduceOpExpToKeepInMixedPrecision_reducesum_without_exp) {
    // ReduceSum without Exp is not a precision sensitive case
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
    auto reduce_sum_1 = make_shared<ReduceSum>(input_1, reduction_axes);

    auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
    auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
    auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
    auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

    model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkReduceOpExpToKeepInMixedPrecision_reducesum_exp_through_unsqueeze) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
        disable_fp16_compression(exp_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(factor_const_decompressed);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(unsqueeze_1);
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

// check marking of Division with eps
TEST(TransformationTests, MarkDivWithEps) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);
        disable_fp16_compression(divide);
        disable_fp16_compression(add);
        disable_fp16_compression(eps_const);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);

    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_PowWithNegativeExponent) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto pow_exp_const = Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<Power>(add, pow_exp_const);
        auto mul = std::make_shared<Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto pow_exp_const = Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<Power>(add, pow_exp_const);
        auto mul = std::make_shared<Multiply>(input_1, pow);
        disable_fp16_compression(mul);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(add);
        disable_fp16_compression(pow_exp_const);
        disable_fp16_compression(pow);

        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_PowWithPositiveExponent) {
    // graph should be left unchanged because of the positive z in pow
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
    auto add = std::make_shared<Add>(input_2, eps_const);
    auto pow_exp_const = Constant::create(element::f32, Shape{1}, {1.77});
    auto pow = std::make_shared<Power>(add, pow_exp_const);
    auto mul = std::make_shared<Multiply>(input_1, pow);

    model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_MinimalPatternUnchanged) {
    // if eps_value is greater than normalized_fp16_min then leave graph unchanged
    const float eps_value = 0.0001f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
    auto add = std::make_shared<Add>(input_2, eps_const);
    auto divide = std::make_shared<Divide>(input_1, add);

    model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkFloatingPointRange) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto begin = Constant::create(element::i64, Shape{}, {0});
        auto step = Constant::create(element::i64, Shape{}, {1});

        auto end = make_shared<Parameter>(element::i64, Shape{});

        auto range_1 = make_shared<op::v4::Range>(begin, end, step, element::f32);
        auto range_2 = make_shared<op::v4::Range>(begin, end, step, element::f32);

        auto convert_1 = make_shared<Convert>(range_1, element::i64);
        auto convert_2 = make_shared<Convert>(convert_1, element::f32);

        auto unsqueeze_const = Constant::create(element::i64, Shape{2}, {-1, 1});
        auto unsqueeze = make_shared<Unsqueeze>(range_2, unsqueeze_const);

        auto greater = make_shared<Greater>(convert_2, unsqueeze);
        auto convert = make_shared<Convert>(greater, element::f32);

        auto multiply_const = Constant::create(element::f32, Shape{}, {1.f});
        auto multiply = make_shared<Multiply>(convert, multiply_const);

        model = make_shared<Model>(NodeVector{convert}, ParameterVector{end});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto begin = Constant::create(element::i64, Shape{}, {0});
        auto step = Constant::create(element::i64, Shape{}, {1});

        auto end = make_shared<Parameter>(element::i64, Shape{});

        auto range_1 = make_shared<op::v4::Range>(begin, end, step, element::f32);
        auto range_2 = make_shared<op::v4::Range>(begin, end, step, element::f32);

        auto convert_1 = make_shared<Convert>(range_1, element::i64);
        auto convert_2 = make_shared<Convert>(convert_1, element::f32);

        auto unsqueeze_const = Constant::create(element::i64, Shape{2}, {-1, 1});
        auto unsqueeze = make_shared<Unsqueeze>(range_2, unsqueeze_const);

        auto greater = make_shared<Greater>(convert_2, unsqueeze);
        auto convert = make_shared<Convert>(greater, element::f32);

        auto multiply_const = Constant::create(element::f32, Shape{}, {1.f});
        auto multiply = make_shared<Multiply>(convert, multiply_const);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(range_1);
        disable_fp16_compression(range_2);
        disable_fp16_compression(convert_1);
        disable_fp16_compression(convert_2);
        disable_fp16_compression(unsqueeze);
        disable_fp16_compression(greater);
        disable_fp16_compression(convert);

        model_ref = make_shared<Model>(NodeVector{convert}, ParameterVector{end});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_InL2NormWithSqrtAndWithMax) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(max);
        auto divide = std::make_shared<Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(max);
        auto divide = std::make_shared<Divide>(input, sqrt);
        disable_fp16_compression(divide);

        disable_fp16_compression(exp);
        disable_fp16_compression(pow);
        disable_fp16_compression(reduce_sum);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(max);
        disable_fp16_compression(sqrt);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_InL2NormWithSqrtAndWithAdd) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(add);
        auto divide = std::make_shared<Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(add);
        auto divide = std::make_shared<Divide>(input, sqrt);
        disable_fp16_compression(divide);

        disable_fp16_compression(exp);
        disable_fp16_compression(pow);
        disable_fp16_compression(reduce_sum);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(add);
        disable_fp16_compression(sqrt);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_disable_for_quantized_nodes_1) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // despite there are sensitive Exp->ReduceSum nodes, but because of the FQ they will
    // be inferred in int8 therefore no need to mark them: model and model_ref should match
    auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto exp_1 = make_shared<opset10::Exp>(input_1);

    auto in_low = op::v0::Constant::create(element::f32, Shape{}, {0.f});
    auto in_high = op::v0::Constant::create(element::f32, Shape{}, {5.f});
    auto out_low = op::v0::Constant::create(element::f32, Shape{}, {2.f});
    auto out_high = op::v0::Constant::create(element::f32, Shape{}, {4.f});
    auto fq_1 = make_shared<opset10::FakeQuantize>(exp_1, in_low, in_high, out_low, out_high, 256);

    auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
    auto reduce_sum_1 = make_shared<opset10::ReduceSum>(fq_1, reduction_axes);

    auto fq_2 = make_shared<opset10::FakeQuantize>(reduce_sum_1, in_low, in_high, out_low, out_high, 256);
    auto matmul_1 = make_shared<opset10::MatMul>(fq_2, input_2);

    model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_disable_for_quantized_nodes_2) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // despite there are sensitive Exp->ReduceSum nodes, but because of the FQ they will
    // be inferred in int8 therefore no need to mark them: model and model_ref should match
    auto input_1 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto input_2 = make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto exp_1 = make_shared<opset10::Exp>(input_1);

    auto in_low = op::v0::Constant::create(element::f32, Shape{}, {0.f});
    auto in_high = op::v0::Constant::create(element::f32, Shape{}, {5.f});
    auto out_low = op::v0::Constant::create(element::f32, Shape{}, {2.f});
    auto out_high = op::v0::Constant::create(element::f32, Shape{}, {4.f});
    auto fq_1 = make_shared<opset10::FakeQuantize>(exp_1, in_low, in_high, out_low, out_high, 256);

    auto unsqueeze_axes = opset10::Constant::create(element::i64, Shape{1}, {1});
    auto unsqueeze_1 = make_shared<opset10::Unsqueeze>(fq_1, unsqueeze_axes);

    auto reduction_axes = opset10::Constant::create(element::i64, Shape{1}, {-1});
    auto reduce_sum_1 = make_shared<opset10::ReduceSum>(unsqueeze_1, reduction_axes);

    auto fq_2 = make_shared<opset10::FakeQuantize>(reduce_sum_1, in_low, in_high, out_low, out_high, 256);
    auto matmul_1 = make_shared<opset10::MatMul>(fq_2, input_2);

    model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}
