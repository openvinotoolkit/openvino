// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/mark_exp_reduceop_to_keep_in_mixed_precision.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "transformations/rt_info/reduceop_path.hpp"

using namespace testing;
using namespace ov;
using namespace std;
using namespace ov::opset10;

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

        manager.register_pass<pass::MarkExpReduceOpToKeepInMixedPrecision>();
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
        mark_reduceop_path(exp_1);
        mark_reduceop_path(reduce_sum_1);
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
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

        manager.register_pass<pass::MarkExpReduceOpToKeepInMixedPrecision>();
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
        mark_reduceop_path(exp_1);
        mark_reduceop_path(reduce_mean_1);
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkReduceOpExpToKeepInMixedPrecision_reducesum_without_exp) {
    // ReduceSum without Exp is not a precision sensitive case
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(input_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkExpReduceOpToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(input_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        mark_reduceop_path(reduce_sum_1);
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
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

        manager.register_pass<pass::MarkExpReduceOpToKeepInMixedPrecision>();
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
        mark_reduceop_path(exp_1);
        mark_reduceop_path(unsqueeze_1);
        mark_reduceop_path(reduce_sum_1);
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}
