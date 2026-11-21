// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <unordered_set>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "openvino/opsets/opset2_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/fp16_compression/mark_subgraphs_to_keep_in_mixed_precision.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace testing;
using namespace ov;
using namespace std;
using namespace ov::opset10;

namespace v0 = ov::op::v0;
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

        model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});

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
        disable_conversion(exp_1, element::f16);
        disable_conversion(reduce_sum_1, element::f16);
        disable_conversion(mul_1, element::f16);
        disable_conversion(factor_const, element::f16);
        disable_conversion(factor_const_decompressed, element::f16);

        model_ref = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
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

        model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});

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
        disable_conversion(exp_1, element::f16);
        disable_conversion(reduce_mean_1, element::f16);
        disable_conversion(mul_1, element::f16);
        disable_conversion(factor_const, element::f16);
        disable_conversion(factor_const_decompressed, element::f16);

        model_ref = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
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

    model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
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

        model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});

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
        disable_conversion(exp_1, ov::element::f16);
        disable_conversion(reduce_sum_1, ov::element::f16);
        disable_conversion(mul_1, ov::element::f16);
        disable_conversion(unsqueeze_1, ov::element::f16);
        disable_conversion(factor_const, ov::element::f16);
        disable_conversion(factor_const_decompressed, ov::element::f16);

        model_ref = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
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

        model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});

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
        disable_conversion(exp_1, ov::element::f16);
        disable_conversion(reduce_sum_1, ov::element::f16);
        disable_conversion(mul_1, ov::element::f16);
        disable_conversion(add_1, ov::element::f16);
        disable_conversion(addition_const, ov::element::f16);
        disable_conversion(factor_const, ov::element::f16);
        disable_conversion(factor_const_decompressed, ov::element::f16);

        model_ref = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
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

        model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2, input_3, input_4});

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
        disable_conversion(mul_1, ov::element::f16);
        disable_conversion(mul_2, ov::element::f16);
        disable_conversion(mul_3, ov::element::f16);
        disable_conversion(mul_4, ov::element::f16);
        disable_conversion(mul_5, ov::element::f16);
        disable_conversion(unsqueeze_1, ov::element::f16);
        disable_conversion(unsqueeze_2, ov::element::f16);
        disable_conversion(unsqueeze_3, ov::element::f16);
        disable_conversion(reduce_sum_1, ov::element::f16);
        disable_conversion(reduce_sum_2, ov::element::f16);
        disable_conversion(reduce_sum_3, ov::element::f16);
        disable_conversion(reduce_sum_4, ov::element::f16);
        disable_conversion(exp_1, ov::element::f16);
        disable_conversion(exp_2, ov::element::f16);
        disable_conversion(tile, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(add_1, ov::element::f16);
        disable_conversion(broadcast, ov::element::f16);
        disable_conversion(div_1, ov::element::f16);

        disable_conversion(factor_1, ov::element::f16);
        disable_conversion(factor_2, ov::element::f16);

        disable_conversion(broadcast_to_shape, ov::element::f16);
        disable_conversion(tile_shape, ov::element::f16);
        disable_conversion(const_unsqueeze_1, ov::element::f16);
        disable_conversion(const_unsqueeze_2, ov::element::f16);
        disable_conversion(const_unsqueeze_3, ov::element::f16);

        model_ref = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2, input_3, input_4});
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
        model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input_1, input_2});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);
        disable_conversion(divide, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(add, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input_1, input_2});
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
        model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input_1, input_2});

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
        disable_conversion(divide, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(convert_eps, ov::element::f16);
        disable_conversion(add, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input_1, input_2});
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

        model = std::make_shared<Model>(OutputVector{mul}, ParameterVector{input_1, input_2});

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
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(add, ov::element::f16);
        disable_conversion(pow_exp_const, ov::element::f16);
        disable_conversion(pow, ov::element::f16);
        disable_conversion(mul, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{mul}, ParameterVector{input_1, input_2});
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

    model = std::make_shared<Model>(OutputVector{mul}, ParameterVector{input_1, input_2});
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

    model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input_1, input_2});
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

        model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});

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
        disable_conversion(exp, ov::element::f16);
        disable_conversion(pow, ov::element::f16);
        disable_conversion(reduce_sum, ov::element::f16);
        disable_conversion(max, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(sqrt, ov::element::f16);
        disable_conversion(divide, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});
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

        model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});

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
        disable_conversion(exp, ov::element::f16);
        disable_conversion(pow, ov::element::f16);
        disable_conversion(reduce_sum, ov::element::f16);
        disable_conversion(max, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(convert_eps, ov::element::f16);
        disable_conversion(sqrt, ov::element::f16);
        disable_conversion(divide, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});
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

        model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});

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
        disable_conversion(exp, ov::element::f16);
        disable_conversion(pow, ov::element::f16);
        disable_conversion(sqrt, ov::element::f16);
        disable_conversion(reduce_sum, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(add, ov::element::f16);
        disable_conversion(divide, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});
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

        model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});

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

        model_ref = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
        disable_conversion(exp_1, ov::element::f16);
        disable_conversion(mul_1, ov::element::f16);
        disable_conversion(reduce_sum_1, ov::element::f16);
        disable_conversion(factor_const_decompressed, ov::element::f16);
        disable_conversion(factor_const, ov::element::f16);
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

        model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});

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

        model_ref = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
        disable_conversion(exp_1, ov::element::f16);
        disable_conversion(mul_1, ov::element::f16);
        disable_conversion(reduce_mean_1, ov::element::f16);
        disable_conversion(factor_const_decompressed, ov::element::f16);
        disable_conversion(factor_const, ov::element::f16);
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

    model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
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

        model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});

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

        model_ref = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
        disable_conversion(exp_1, ov::element::f16);
        disable_conversion(mul_1, ov::element::f16);
        disable_conversion(reduce_sum_1, ov::element::f16);
        disable_conversion(factor_const_decompressed, ov::element::f16);
        disable_conversion(factor_const, ov::element::f16);
        disable_conversion(unsqueeze_1, ov::element::f16);
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

        model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);
        disable_conversion(divide, ov::element::f16);
        disable_conversion(add, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input_1, input_2});
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

        model = std::make_shared<Model>(OutputVector{mul}, ParameterVector{input_1, input_2});
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
        disable_conversion(mul, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(add, ov::element::f16);
        disable_conversion(pow_exp_const, ov::element::f16);
        disable_conversion(pow, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{mul}, ParameterVector{input_1, input_2});
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

    model = std::make_shared<Model>(OutputVector{mul}, ParameterVector{input_1, input_2});
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

    model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input_1, input_2});
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

        model = make_shared<Model>(OutputVector{convert}, ParameterVector{end});

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
        disable_conversion(begin, ov::element::f16);
        disable_conversion(end, ov::element::f16);
        disable_conversion(step, ov::element::f16);
        disable_conversion(range_1, ov::element::f16);
        disable_conversion(range_2, ov::element::f16);
        disable_conversion(convert_1, ov::element::f16);
        disable_conversion(convert_2, ov::element::f16);
        disable_conversion(unsqueeze, ov::element::f16);
        disable_conversion(greater, ov::element::f16);
        disable_conversion(convert, ov::element::f16);

        model_ref = make_shared<Model>(OutputVector{convert}, ParameterVector{end});
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

        model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});
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
        disable_conversion(divide, ov::element::f16);

        disable_conversion(exp, ov::element::f16);
        disable_conversion(pow, ov::element::f16);
        disable_conversion(reduce_sum, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(max, ov::element::f16);
        disable_conversion(sqrt, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});
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

        model = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});
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
        disable_conversion(divide, ov::element::f16);

        disable_conversion(exp, ov::element::f16);
        disable_conversion(pow, ov::element::f16);
        disable_conversion(reduce_sum, ov::element::f16);
        disable_conversion(eps_const, ov::element::f16);
        disable_conversion(add, ov::element::f16);
        disable_conversion(sqrt, ov::element::f16);

        model_ref = std::make_shared<Model>(OutputVector{divide}, ParameterVector{input});
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

    model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
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

    model = make_shared<Model>(OutputVector{matmul_1}, ParameterVector{input_1, input_2});
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

TEST_F(TransformationTestsF, MarkRandomUniformAsPrecisionSensitive) {
    auto param = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{2});
    auto random_uniform = std::make_shared<ov::op::v8::RandomUniform>(param,
                                                                      v0::Constant::create(element::f32, {}, {0}),
                                                                      v0::Constant::create(element::f32, {}, {1}),
                                                                      element::f32);
    auto less = std::make_shared<ov::op::v1::Less>(random_uniform, v0::Constant::create(element::f32, {1, 1}, {0.5}));
    auto res = std::make_shared<v0::Result>(less);

    model = std::make_shared<ov::Model>(OutputVector{res}, ParameterVector{param});

    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};

    type_to_fuse_map empty_fuse_map;

    model_ref = model->clone();
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, empty_fuse_map, true, false, true);
}

namespace {

void compare_marks(const shared_ptr<Model>& model, const shared_ptr<Model>& model_ref) {
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

// sum(hidden * mask, 1) / mask.sum(1): masked mean pooling without eps (HF ModernBERT), preceded by residual Adds
shared_ptr<Model> make_masked_mean_pooling(const PartialShape& hidden_shape, bool reference) {
    auto hidden = make_shared<Parameter>(element::f32, hidden_shape);
    auto residual = make_shared<Parameter>(element::f32, hidden_shape);
    auto add_1 = make_shared<Add>(hidden, residual);
    auto add_2 = make_shared<Add>(add_1, residual);
    auto weights = Constant::create(element::f32, Shape{8, 8}, vector<float>(64, 0.5f));
    auto other_branch = make_shared<MatMul>(add_2, weights);

    auto mask = make_shared<Parameter>(element::i64, PartialShape{hidden_shape[0], hidden_shape[1]});
    auto mask_unsqueezed = make_shared<Unsqueeze>(mask, Constant::create(element::i64, Shape{1}, {-1}));
    auto mask_converted = make_shared<Convert>(mask_unsqueezed, element::f32);
    auto masked = make_shared<Multiply>(add_2, mask_converted);
    auto sum = make_shared<ReduceSum>(masked, Constant::create(element::i64, Shape{1}, {1}), false);
    auto count = make_shared<ReduceSum>(mask, Constant::create(element::i64, Shape{1}, {1}), true);
    auto count_converted = make_shared<Convert>(count, element::f32);
    count_converted->set_friendly_name("count_converted");
    auto mean = make_shared<Divide>(sum, count_converted);
    auto classifier = make_shared<MatMul>(mean, weights);

    if (reference) {
        for (const auto& node : NodeVector{masked, sum, count_converted, mean}) {
            disable_conversion(node, element::f16);
        }
    }
    return make_shared<Model>(OutputVector{classifier, other_branch}, ParameterVector{hidden, residual, mask});
}

}  // namespace

TEST(TransformationTests, MarkSumDivCountToKeepInMixedPrecision_MaskedMeanPooling) {
    for (const auto& shape : {PartialShape{2, 16, 8}, PartialShape{-1, -1, 8}}) {
        auto model = make_masked_mean_pooling(shape, false);
        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
        compare_marks(model, make_masked_mean_pooling(shape, true));
    }
}

// sum(hidden * w, 1) / w.sum(1) with float weights: the weighted product is kept in fp32 as well
TEST(TransformationTests, MarkSumDivCountToKeepInMixedPrecision_FloatWeights) {
    auto build = [](bool reference) {
        auto hidden = make_shared<Parameter>(element::f32, PartialShape{2, -1, 8});
        auto residual = make_shared<Parameter>(element::f32, PartialShape{2, -1, 8});
        auto add = make_shared<Add>(hidden, residual);
        auto weights = make_shared<Parameter>(element::f32, PartialShape{2, -1, 1});
        auto weighted = make_shared<Multiply>(add, weights);
        auto sum = make_shared<ReduceSum>(weighted, Constant::create(element::i64, Shape{1}, {1}), false);
        auto count = make_shared<ReduceSum>(weights, Constant::create(element::i64, Shape{1}, {1}), false);
        auto mean = make_shared<Divide>(sum, count);
        if (reference) {
            for (const auto& node : NodeVector{weighted, sum, count, mean}) {
                disable_conversion(node, element::f16);
            }
        }
        return make_shared<Model>(OutputVector{mean}, ParameterVector{hidden, residual, weights});
    };
    auto model = build(false);
    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);
    compare_marks(model, build(true));
}

// wav2vec2 heads: the integer count is reshaped to [B, 1] before the Convert
TEST(TransformationTests, MarkSumDivCountToKeepInMixedPrecision_ReshapedIntegerCount) {
    auto build = [](bool reference) {
        auto hidden = make_shared<Parameter>(element::f32, PartialShape{2, -1, 8});
        auto mask = make_shared<Parameter>(element::i64, PartialShape{2, -1});
        auto mask_converted =
            make_shared<Convert>(make_shared<Unsqueeze>(mask, Constant::create(element::i64, Shape{1}, {-1})),
                                 element::f32);
        auto masked = make_shared<Multiply>(hidden, mask_converted);
        auto sum = make_shared<ReduceSum>(masked, Constant::create(element::i64, Shape{1}, {1}), false);
        auto count = make_shared<ReduceSum>(mask, Constant::create(element::i64, Shape{1}, {1}), false);
        auto target_shape = Constant::create(element::i64, Shape{2}, {-1, 1});
        auto count_reshaped = make_shared<Reshape>(count, target_shape, false);
        auto count_converted = make_shared<Convert>(count_reshaped, element::f32);
        auto mean = make_shared<Divide>(sum, count_converted);
        if (reference) {
            for (const auto& node : NodeVector{masked, sum, count_converted, mean}) {
                disable_conversion(node, element::f16);
            }
            // marked by MarkPrecisionSensitiveShapeOfSubgraphs: the target shape is a precision sensitive input
            disable_conversion(target_shape, element::f16);
        }
        return make_shared<Model>(OutputVector{mean}, ParameterVector{hidden, mask});
    };
    auto model = build(false);
    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);
    compare_marks(model, build(true));
}

// the converted count is unsqueezed to [B, 1]: the Unsqueeze is on the count path, kept in fp32 as well
TEST(TransformationTests, MarkSumDivCountToKeepInMixedPrecision_UnsqueezedConvertedCount) {
    auto build = [](bool reference) {
        auto hidden = make_shared<Parameter>(element::f32, PartialShape{2, -1, 8});
        auto mask = make_shared<Parameter>(element::i64, PartialShape{2, -1});
        auto mask_converted =
            make_shared<Convert>(make_shared<Unsqueeze>(mask, Constant::create(element::i64, Shape{1}, {-1})),
                                 element::f32);
        auto masked = make_shared<Multiply>(hidden, mask_converted);
        auto sum = make_shared<ReduceSum>(masked, Constant::create(element::i64, Shape{1}, {1}), false);
        auto count = make_shared<ReduceSum>(mask, Constant::create(element::i64, Shape{1}, {1}), false);
        auto count_converted = make_shared<Convert>(count, element::f32);
        auto unsqueeze_axis = Constant::create(element::i64, Shape{1}, {-1});
        auto count_unsqueezed = make_shared<Unsqueeze>(count_converted, unsqueeze_axis);
        auto mean = make_shared<Divide>(sum, count_unsqueezed);
        if (reference) {
            for (const auto& node : NodeVector{masked, sum, count_converted, count_unsqueezed, mean}) {
                disable_conversion(node, element::f16);
            }
        }
        return make_shared<Model>(OutputVector{mean}, ParameterVector{hidden, mask});
    };
    auto model = build(false);
    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);
    compare_marks(model, build(true));
}

// masked RMSNorm x / sqrt(sum(x * x * mask) / sum(mask) + eps): the eps pattern still keeps the whole norm in fp32,
// including x * x above the masked mean
TEST(TransformationTests, MarkSumDivCountToKeepInMixedPrecision_MaskedRMSNormKeepsSquareMarked) {
    auto x = make_shared<Parameter>(element::f32, PartialShape{2, -1, 8});
    auto mask = make_shared<Parameter>(element::f32, PartialShape{2, -1, 1});
    auto square = make_shared<Multiply>(x, x);
    auto masked = make_shared<Multiply>(square, mask);
    auto sum = make_shared<ReduceSum>(masked, Constant::create(element::i64, Shape{1}, {1}), true);
    auto count = make_shared<ReduceSum>(mask, Constant::create(element::i64, Shape{1}, {1}), true);
    auto mean = make_shared<Divide>(sum, count);
    auto eps = Constant::create(element::f32, Shape{}, {1e-6f});
    auto add_eps = make_shared<Add>(mean, eps);
    auto root = make_shared<Sqrt>(add_eps);
    auto normalized = make_shared<Divide>(x, root);
    auto model = make_shared<Model>(OutputVector{normalized}, ParameterVector{x, mask});

    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    for (const auto& node : NodeVector{square, masked, sum, mean}) {
        EXPECT_TRUE(is_conversion_disabled(node, element::f16)) << node;
    }
}

// the masking product has another consumer: it keeps its precision, only the pooling itself is kept in fp32
TEST(TransformationTests, MarkSumDivCountToKeepInMixedPrecision_SharedProductIsNotMarked) {
    auto build = [](bool reference) {
        auto hidden = make_shared<Parameter>(element::f32, PartialShape{2, -1, 8});
        auto weights = make_shared<Parameter>(element::f32, PartialShape{2, -1, 1});
        auto weighted = make_shared<Multiply>(hidden, weights);
        auto other = make_shared<Relu>(weighted);
        auto sum = make_shared<ReduceSum>(weighted, Constant::create(element::i64, Shape{1}, {1}), false);
        auto count = make_shared<ReduceSum>(weights, Constant::create(element::i64, Shape{1}, {1}), false);
        auto mean = make_shared<Divide>(sum, count);
        if (reference) {
            for (const auto& node : NodeVector{sum, count, mean}) {
                disable_conversion(node, element::f16);
            }
        }
        return make_shared<Model>(OutputVector{mean, other}, ParameterVector{hidden, weights});
    };
    auto model = build(false);
    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);
    compare_marks(model, build(true));
}

// the count comes from the shape, not from a ReduceSum: e.g. a hand-written mean, not marked
TEST(TransformationTests, MarkSumDivCountToKeepInMixedPrecision_ShapeOfDivisorIsNotMarked) {
    auto build = [] {
        auto hidden = make_shared<Parameter>(element::f32, PartialShape{2, -1, 8});
        auto sum = make_shared<ReduceSum>(hidden, Constant::create(element::i64, Shape{1}, {1}), false);
        auto length = make_shared<Gather>(make_shared<ShapeOf>(hidden),
                                          Constant::create(element::i64, Shape{1}, {1}),
                                          Constant::create(element::i64, Shape{}, {0}));
        auto mean = make_shared<Divide>(sum, make_shared<Convert>(length, element::f32));
        return make_shared<Model>(OutputVector{mean}, ParameterVector{hidden});
    };
    auto model = build();
    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);
    compare_marks(model, build());
}

// the sum and the count are taken over different axes: not a mean, not marked
TEST(TransformationTests, MarkSumDivCountToKeepInMixedPrecision_DifferentAxesAreNotMarked) {
    auto build = [] {
        auto hidden = make_shared<Parameter>(element::f32, PartialShape{2, 16, 8});
        auto mask = make_shared<Parameter>(element::f32, PartialShape{2, 16, 8});
        auto sum = make_shared<ReduceSum>(hidden, Constant::create(element::i64, Shape{1}, {1}), true);
        auto count = make_shared<ReduceSum>(mask, Constant::create(element::i64, Shape{1}, {2}), true);
        auto ratio = make_shared<Divide>(sum, count);
        return make_shared<Model>(OutputVector{ratio}, ParameterVector{hidden, mask});
    };
    auto model = build();
    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);
    compare_marks(model, build());
}

namespace {

// the precisions after ConvertPrecision of a model built by make_masked_mean_pooling: the pooling is computed in fp32,
// its producers and consumers keep f16
void check_masked_mean_pooling_precisions(const shared_ptr<Model>& model) {
    size_t checked = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (is_type_any_of<Divide, ReduceSum, Multiply>(node) && node->get_output_element_type(0).is_real()) {
            EXPECT_EQ(node->get_output_element_type(0), element::f32) << node;
            checked++;
        } else if (is_type<Add>(node)) {
            EXPECT_EQ(node->get_output_element_type(0), element::f16) << node;
            checked++;
        } else if (is_type<MatMul>(node)) {
            // the classifier after the pooling and the other consumer of the hidden states keep f16
            EXPECT_EQ(node->get_output_element_type(0), element::f16) << node;
            checked++;
        } else if (node->get_friendly_name() == "count_converted") {
            // an integer count above 2048 is not exact in f16
            EXPECT_EQ(node->get_output_element_type(0), element::f32) << node;
            checked++;
        }
    }
    // Divide, the f32 ReduceSum, Multiply, two Adds, two MatMuls and the count Convert
    EXPECT_EQ(checked, 8u);
}

void convert_precision_keep_sensitive(const shared_ptr<Model>& model) {
    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true,
                                                  false);
    manager.run_passes(model);
}

}  // namespace

// with ConvertPrecision as in the CPU plugin: the pooling is computed in fp32, its producers keep f16
TEST(TransformationTests, ConvertPrecision_MaskedMeanPoolingKeptInFP32) {
    auto model = make_masked_mean_pooling(PartialShape{-1, -1, 8}, false);
    convert_precision_keep_sensitive(model);
    check_masked_mean_pooling_precisions(model);
}

// ConvertPrecision marks the nodes of an If/Loop body twice: in the run on the parent model, which visits the bodies,
// and in the run on the body. The masked means of a body are kept in fp32 without spreading to the residual Adds.
TEST(TransformationTests, ConvertPrecision_MaskedMeanPoolingInIfBodyKeptInFP32) {
    const auto then_body = make_masked_mean_pooling(PartialShape{2, 16, 8}, false);
    const auto else_body = make_masked_mean_pooling(PartialShape{2, 16, 8}, false);
    auto if_op = make_shared<If>(Constant::create(element::boolean, Shape{1}, {true}));
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    ParameterVector params;
    for (size_t i = 0; i < then_body->get_parameters().size(); ++i) {
        const auto& body_param = then_body->get_parameters()[i];
        params.push_back(make_shared<Parameter>(body_param->get_element_type(), body_param->get_partial_shape()));
        if_op->set_input(params.back(), body_param, else_body->get_parameters()[i]);
    }
    OutputVector outputs;
    for (size_t i = 0; i < then_body->get_results().size(); ++i) {
        outputs.push_back(if_op->set_output(then_body->get_results()[i], else_body->get_results()[i]));
    }
    const auto model = make_shared<Model>(outputs, params);

    convert_precision_keep_sensitive(model);

    check_masked_mean_pooling_precisions(then_body);
    check_masked_mean_pooling_precisions(else_body);
}

// the same in the body of a Loop
TEST(TransformationTests, ConvertPrecision_MaskedMeanPoolingInLoopBodyKeptInFP32) {
    const auto pooling = make_masked_mean_pooling(PartialShape{2, 16, 8}, false);
    auto results = pooling->get_results();
    results.push_back(make_shared<Result>(Constant::create(element::boolean, Shape{}, {true})));
    const auto body = make_shared<Model>(results, pooling->get_parameters());
    auto loop = make_shared<Loop>(Constant::create(element::i64, Shape{}, {1}),
                                  Constant::create(element::boolean, Shape{}, {true}));
    loop->set_function(body);
    loop->set_special_body_ports({-1, static_cast<int64_t>(results.size() - 1)});
    ParameterVector params;
    for (const auto& body_param : body->get_parameters()) {
        params.push_back(make_shared<Parameter>(body_param->get_element_type(), body_param->get_partial_shape()));
        loop->set_invariant_input(body_param, params.back());
    }
    OutputVector outputs;
    for (size_t i = 0; i + 1 < results.size(); ++i) {
        outputs.push_back(loop->get_iter_value(results[i], -1));
    }
    const auto model = make_shared<Model>(outputs, params);

    convert_precision_keep_sensitive(model);

    check_masked_mean_pooling_precisions(body);
}

namespace {

// 1 / Maximum(1 - Convert(Cos(x), f32), eps): the marks of the eps-guarded division propagate up through the Convert
// only if it does not change the type (e.g. a Cast to the same type, or the boundary of DisableFP16CompForRopeAngles)
shared_ptr<Model> make_eps_division_after_convert(const element::Type& input_type, bool reference) {
    auto input = make_shared<Parameter>(input_type, PartialShape{2, -1, 8});
    auto cos = make_shared<Cos>(input);
    auto convert = make_shared<Convert>(cos, element::f32);
    auto one = Constant::create(element::f32, Shape{}, {1.0f});
    auto one_minus = make_shared<Subtract>(one, convert);
    auto eps = Constant::create(element::f32, Shape{}, {1.0e-12f});
    auto clamped = make_shared<Maximum>(one_minus, eps);
    auto numerator = Constant::create(element::f32, Shape{}, {1.0f});
    auto division = make_shared<Divide>(numerator, clamped);
    if (reference) {
        for (const auto& node : NodeVector{one, one_minus, eps, clamped, numerator, division}) {
            disable_conversion(node, element::f16);
        }
        if (input_type == element::f32) {
            disable_conversion(convert, element::f16);
            disable_conversion(cos, element::f16);
        }
    }
    return make_shared<Model>(OutputVector{division}, ParameterVector{input});
}

}  // namespace

// a Convert that keeps the type is a no-op: the marks propagate up through it to Cos
TEST(TransformationTests, MarkSugraphsToKeepInMixedPrecision_PropagateUpThroughNoOpConvert) {
    auto model = make_eps_division_after_convert(element::f32, false);
    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);
    compare_marks(model, make_eps_division_after_convert(element::f32, true));
}

// a Convert to f32 of a non-constant f16 (or not yet inferred) input: the propagation stops at it, Cos is not marked
TEST(TransformationTests, MarkSugraphsToKeepInMixedPrecision_PropagateUpStopsAtTypeChangingConvert) {
    for (const auto& input_type : {element::f16, element::dynamic}) {
        SCOPED_TRACE(input_type);
        auto model = make_eps_division_after_convert(input_type, false);
        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
        compare_marks(model, make_eps_division_after_convert(input_type, true));
    }
}
