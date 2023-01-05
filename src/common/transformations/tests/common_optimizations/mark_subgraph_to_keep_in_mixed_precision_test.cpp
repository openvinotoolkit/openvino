// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/common_optimizations/mark_subgraphs_to_keep_in_mixed_precision.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/reduceop_path.hpp"

using namespace testing;
using namespace ov;
using namespace std;

TEST(TransformationTests, keep_precission_sensitive_fp32_1) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp_1);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(factor_const_decompressed);

        // marking for Exp->ReduceSum path
        mark_reduceop_path(exp_1);
        mark_reduceop_path(reduce_sum_1);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, keep_precission_sensitive_fp32_with_reducemean) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_mean_1 = make_shared<opset8::ReduceMean>(exp_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_mean_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp_1);
        disable_fp16_compression(reduce_mean_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(factor_const_decompressed);

        // marking for Exp->ReduceSum path
        mark_reduceop_path(exp_1);
        mark_reduceop_path(reduce_mean_1);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, MarkSugraphsToKeepInMixedPrecision_reducesum_without_exp) {
    // ReduceSum without Exp is not a precision sensitive case,
    // no nodes should be marked with disable_fp16_compression
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(input_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(input_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        mark_reduceop_path(reduce_sum_1);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, keep_precission_sensitive_fp32_2) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<opset8::Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<opset8::Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp_1);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(unsqueeze_1);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(factor_const_decompressed);

        // marking for Exp->ReduceSum path
        mark_reduceop_path(exp_1);
        mark_reduceop_path(unsqueeze_1);
        mark_reduceop_path(reduce_sum_1);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, keep_precission_sensitive_fp32_3) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});

        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Add>(reduce_sum_1, addition_const);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(add_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});

        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Add>(reduce_sum_1, addition_const);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(add_1, factor_const_decompressed);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp_1);
        disable_fp16_compression(reduce_sum_1);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(add_1);
        disable_fp16_compression(addition_const);
        disable_fp16_compression(factor_const);
        disable_fp16_compression(factor_const_decompressed);

        // marking for Exp->ReduceSum path
        mark_reduceop_path(exp_1);
        mark_reduceop_path(reduce_sum_1);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    //    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, keep_precission_sensitive_fp32_4) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto mvn_1 = make_shared<opset8::MVN>(input_1, reduction_axes, true, 1.e-8, op::MVNEpsMode::INSIDE_SQRT);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Unsqueeze>(mvn_1, addition_const);
        auto matmul_1 = make_shared<opset8::MatMul>(add_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto mvn_1 = make_shared<opset8::MVN>(input_1, reduction_axes, true, 1.e-8, op::MVNEpsMode::INSIDE_SQRT);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Unsqueeze>(mvn_1, addition_const);
        auto matmul_1 = make_shared<opset8::MatMul>(add_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(addition_const);
        disable_fp16_compression(add_1);
        disable_fp16_compression(mvn_1);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, keep_precission_sensitive_fp32_5) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto normalizel2_1 = make_shared<opset8::NormalizeL2>(input_1, reduction_axes, 1.e-8, ov::op::EpsMode::MAX);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Unsqueeze>(normalizel2_1, addition_const);
        auto matmul_1 = make_shared<opset8::MatMul>(add_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto normalizel2_1 = make_shared<opset8::NormalizeL2>(input_1, reduction_axes, 1.e-8, ov::op::EpsMode::MAX);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Unsqueeze>(normalizel2_1, addition_const);
        auto matmul_1 = make_shared<opset8::MatMul>(add_1, input_2);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(addition_const);
        disable_fp16_compression(add_1);
        disable_fp16_compression(normalizel2_1);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, keep_precission_sensitive_fp32_5__) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto normalizel2_1 = make_shared<opset8::NormalizeL2>(input_1, reduction_axes, 1.e-8, ov::op::EpsMode::MAX);
        auto matmul_1 = make_shared<opset8::MatMul>(normalizel2_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto normalizel2_1 = make_shared<opset8::NormalizeL2>(input_1, reduction_axes, 1.e-8, ov::op::EpsMode::MAX);
        auto matmul_1 = make_shared<opset8::MatMul>(normalizel2_1, input_2);

        disable_fp16_compression(normalizel2_1);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, keep_precission_sensitive_fp32_300) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // subgraph from t2t-vit-7
    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_3 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3136, 64, 1});
        auto input_4 = make_shared<opset8::Parameter>(element::f32, Shape{128, 64});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto exp_2 = make_shared<opset8::Exp>(input_2);

        auto factor_1 = opset8::Constant::create(element::f32, Shape{1}, {0.5});  // add decompression
        auto mul_1 = make_shared<opset8::Multiply>(exp_1, factor_1);
        auto factor_2 = opset8::Constant::create(element::f32, Shape{1}, {0.5});
        auto mul_2 = make_shared<opset8::Multiply>(exp_2, factor_2);

        auto const_unsqueeze_1 = opset8::Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_1 = make_shared<opset8::Reshape>(mul_1, const_unsqueeze_1, false);

        auto const_unsqueeze_2 = opset8::Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_2 = make_shared<opset8::Reshape>(mul_2, const_unsqueeze_1, false);
        auto reduction_axes_1 = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(mul_2, reduction_axes_1, true);
        auto mul_3 = make_shared<opset8::Multiply>(reduce_sum_1, mul_1);
        auto mul_4 = make_shared<opset8::Multiply>(input_3, unsqueeze_2);

        auto reduction_axes_2 = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_2 = make_shared<opset8::ReduceSum>(mul_4, reduction_axes_2);
        auto reduction_axes_3 = opset8::Constant::create(element::i64, Shape{1}, {2});
        auto reduce_sum_3 = make_shared<opset8::ReduceSum>(mul_3, reduction_axes_3, true);

        auto broadcast_to_shape = opset8::Constant::create(element::i64, Shape{3}, {1, 1, 1});
        auto broadcast =
            make_shared<opset8::Broadcast>(reduce_sum_3, broadcast_to_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto tile_shape = opset8::Constant::create(element::i64, Shape{3}, {1, 1, 64});
        auto tile = make_shared<opset8::Tile>(broadcast, tile_shape);
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {1.e-10});
        auto add_1 = make_shared<opset8::Add>(tile, eps_const);

        auto const_unsqueeze_3 = opset8::Constant::create(element::i64, Shape{4}, {1, 1, 64, 32});
        auto unsqueeze_3 = make_shared<opset8::Reshape>(reduce_sum_2, const_unsqueeze_3, false);
        auto mul_5 = make_shared<opset8::Multiply>(unsqueeze_1, unsqueeze_3);

        auto reduction_axes_4 = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_4 = make_shared<opset8::ReduceSum>(mul_5, reduction_axes_4);

        auto div_1 = make_shared<opset8::Divide>(reduce_sum_4, add_1);
        auto matmul_1 = make_shared<opset8::MatMul>(div_1, input_4, false, true);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2, input_3, input_4});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
        pass::VisualizeTree("after.svg").run_on_model(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3136, 32});
        auto input_3 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3136, 64, 1});
        auto input_4 = make_shared<opset8::Parameter>(element::f32, Shape{128, 64});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto exp_2 = make_shared<opset8::Exp>(input_2);

        auto factor_1 = opset8::Constant::create(element::f32, Shape{1}, {0.5});  // add decompression
        auto mul_1 = make_shared<opset8::Multiply>(exp_1, factor_1);
        auto factor_2 = opset8::Constant::create(element::f32, Shape{1}, {0.5});
        auto mul_2 = make_shared<opset8::Multiply>(exp_2, factor_2);

        auto const_unsqueeze_1 = opset8::Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_1 = make_shared<opset8::Reshape>(mul_1, const_unsqueeze_1, false);

        auto const_unsqueeze_2 = opset8::Constant::create(element::i64, Shape{4}, {1, 3136, 1, 32});
        auto unsqueeze_2 = make_shared<opset8::Reshape>(mul_2, const_unsqueeze_2, false);
        auto reduction_axes_1 = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(mul_2, reduction_axes_1, true);
        auto mul_3 = make_shared<opset8::Multiply>(reduce_sum_1, mul_1);
        auto mul_4 = make_shared<opset8::Multiply>(input_3, unsqueeze_2);

        auto reduction_axes_2 = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum_2 = make_shared<opset8::ReduceSum>(mul_4, reduction_axes_2);
        auto reduction_axes_3 = opset8::Constant::create(element::i64, Shape{1}, {2});
        auto reduce_sum_3 = make_shared<opset8::ReduceSum>(mul_3, reduction_axes_3, true);

        auto broadcast_to_shape = opset8::Constant::create(element::i64, Shape{3}, {1, 1, 1});
        auto broadcast =
            make_shared<opset8::Broadcast>(reduce_sum_3, broadcast_to_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto tile_shape = opset8::Constant::create(element::i64, Shape{3}, {1, 1, 64});
        auto tile = make_shared<opset8::Tile>(broadcast, tile_shape);
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {1.e-10});
        auto add_1 = make_shared<opset8::Add>(tile, eps_const);

        auto const_unsqueeze_3 = opset8::Constant::create(element::i64, Shape{4}, {1, 1, 64, 32});
        auto unsqueeze_3 = make_shared<opset8::Reshape>(reduce_sum_2, const_unsqueeze_3, false);
        auto mul_5 = make_shared<opset8::Multiply>(unsqueeze_1, unsqueeze_3);

        auto reduction_axes_4 = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_4 = make_shared<opset8::ReduceSum>(mul_5, reduction_axes_4);

        auto div_1 = make_shared<opset8::Divide>(reduce_sum_4, add_1);
        auto matmul_1 = make_shared<opset8::MatMul>(div_1, input_4, false, true);

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

        // marking for Exp->ReduceSum path
        mark_reduceop_path(mul_1);
        mark_reduceop_path(mul_2);
        mark_reduceop_path(mul_3);
        mark_reduceop_path(mul_4);
        mark_reduceop_path(mul_5);
        mark_reduceop_path(unsqueeze_1);
        mark_reduceop_path(unsqueeze_2);
        mark_reduceop_path(unsqueeze_3);
        mark_reduceop_path(reduce_sum_1);
        mark_reduceop_path(reduce_sum_2);
        mark_reduceop_path(reduce_sum_3);
        mark_reduceop_path(reduce_sum_4);
        mark_reduceop_path(exp_1);
        mark_reduceop_path(exp_2);

        mark_reduceop_path(factor_1);
        mark_reduceop_path(factor_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2, input_3, input_4});
        pass::VisualizeTree("ref.svg").run_on_model(model_ref);
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, DivisionByZeroMinimalPattern) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    const float eps_value = 1.0e-12f;
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset8::Divide>(input_1, add);
        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset8::Divide>(input_1, add);
        disable_fp16_compression(divide);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(add);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, PowWithNegativeExponent) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    const float eps_value = 1.0e-12f;
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(input_2, eps_const);
        auto pow_exp_const = opset8::Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<opset8::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset8::Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(input_2, eps_const);
        auto pow_exp_const = opset8::Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<opset8::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset8::Multiply>(input_1, pow);

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
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, PowWithPositiveExponent) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // graph should be left unchanged
    const float eps_value = 1.0e-12f;
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(input_2, eps_const);
        auto pow_exp_const = opset8::Constant::create(element::f32, Shape{1}, {1.77});
        auto pow = std::make_shared<opset8::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset8::Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(input_2, eps_const);
        auto pow_exp_const = opset8::Constant::create(element::f32, Shape{1}, {1.77});
        auto pow = std::make_shared<opset8::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset8::Multiply>(input_1, pow);

        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, DivisionByZeroMinimalPatternUnchanged) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    // if eps_value is greater than normalized_fp16_min then leave graph unchanged
    const float eps_value = 0.0001f;
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset8::Divide>(input_1, add);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset8::Divide>(input_1, add);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, DivisionByZeroInL2NormWithSqrtAndWithMax) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    const float eps_value = 1.0e-12f;
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset8::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset8::Power>(input, exp);
        auto axes_const = opset8::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset8::ReduceSum>(pow, axes_const);
        auto eps_const = opset8::Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<opset8::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset8::Sqrt>(max);
        auto divide = std::make_shared<opset8::Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset8::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset8::Power>(input, exp);
        auto axes_const = opset8::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset8::ReduceSum>(pow, axes_const);
        auto eps_const = opset8::Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<opset8::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset8::Sqrt>(max);
        auto divide = std::make_shared<opset8::Divide>(input, sqrt);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp);
        disable_fp16_compression(pow);
        disable_fp16_compression(reduce_sum);
        disable_fp16_compression(max);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(sqrt);
        disable_fp16_compression(divide);

        // marking for Exp->ReduceSum path
        mark_reduceop_path(exp);
        mark_reduceop_path(pow);
        mark_reduceop_path(reduce_sum);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, DivisionByZeroInL2NormWithSqrtAndWithAdd) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    const float eps_value = 1.e-12;
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset8::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset8::Power>(input, exp);
        auto axes_const = opset8::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset8::ReduceSum>(pow, axes_const);
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset8::Sqrt>(add);
        auto divide = std::make_shared<opset8::Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset8::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset8::Power>(input, exp);
        auto axes_const = opset8::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset8::ReduceSum>(pow, axes_const);
        auto eps_const = opset8::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset8::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset8::Sqrt>(add);
        auto divide = std::make_shared<opset8::Divide>(input, sqrt);

        // marking nodes to be kept in fp32 for mixed precision
        disable_fp16_compression(exp);
        disable_fp16_compression(pow);
        disable_fp16_compression(sqrt);
        disable_fp16_compression(reduce_sum);
        disable_fp16_compression(eps_const);
        disable_fp16_compression(add);
        disable_fp16_compression(divide);

        // marking for Exp->ReduceSum path
        mark_reduceop_path(exp);
        mark_reduceop_path(pow);
        mark_reduceop_path(reduce_sum);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}
