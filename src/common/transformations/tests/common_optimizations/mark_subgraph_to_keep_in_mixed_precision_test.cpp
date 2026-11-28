// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
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

namespace {
void disable_fp16_conversion(const std::vector<std::shared_ptr<Node>>& nodes) {
    for (const auto& node : nodes) {
        disable_conversion(node, element::f16);
    }
}
}  // namespace

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

TEST(TransformationTests, keep_precission_sensitive_integer_to_float_subgraph) {
    // DeBERTa-like relative position bucket computation:
    // integer positions -> f32 -> elementwise math -> integer buckets.
    // Rounding the f32 chain to f16 can change the result of Ceiling and silently
    // produce a wrong bucket index, so the whole chain must stay in f32.
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    {
        auto relative_position = make_shared<Parameter>(element::i64, Shape{2, 4});
        auto relative_position_f = make_shared<Convert>(relative_position, element::f32);
        auto abs = make_shared<Abs>(relative_position_f);
        auto zero = Constant::create(element::f32, Shape{}, {0});
        auto greater = make_shared<Greater>(abs, zero);
        auto less = make_shared<Less>(abs, zero);
        auto condition = make_shared<LogicalAnd>(greater, less);
        auto select_1 = make_shared<Select>(condition, abs, zero);
        auto two = Constant::create(element::f32, Shape{}, {2});
        auto divide_1 = make_shared<Divide>(select_1, two);
        auto log = make_shared<Log>(divide_1);
        auto divide_2 = make_shared<Divide>(log, two);
        auto multiply_1 = make_shared<Multiply>(divide_2, two);
        auto ceiling = make_shared<Ceiling>(multiply_1);
        auto add = make_shared<Add>(ceiling, zero);
        auto multiply_2 = make_shared<Multiply>(add, two);
        auto select_2 = make_shared<Select>(condition, multiply_2, zero);
        auto bucket = make_shared<Convert>(select_2, element::i64);

        model = make_shared<Model>(OutputVector{bucket}, ParameterVector{relative_position});

        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto relative_position = make_shared<Parameter>(element::i64, Shape{2, 4});
        auto relative_position_f = make_shared<Convert>(relative_position, element::f32);
        auto abs = make_shared<Abs>(relative_position_f);
        auto zero = Constant::create(element::f32, Shape{}, {0});
        auto greater = make_shared<Greater>(abs, zero);
        auto less = make_shared<Less>(abs, zero);
        auto condition = make_shared<LogicalAnd>(greater, less);
        auto select_1 = make_shared<Select>(condition, abs, zero);
        auto two = Constant::create(element::f32, Shape{}, {2});
        auto divide_1 = make_shared<Divide>(select_1, two);
        auto log = make_shared<Log>(divide_1);
        auto divide_2 = make_shared<Divide>(log, two);
        auto multiply_1 = make_shared<Multiply>(divide_2, two);
        auto ceiling = make_shared<Ceiling>(multiply_1);
        auto add = make_shared<Add>(ceiling, zero);
        auto multiply_2 = make_shared<Multiply>(add, two);
        auto select_2 = make_shared<Select>(condition, multiply_2, zero);
        auto bucket = make_shared<Convert>(select_2, element::i64);

        disable_fp16_conversion({relative_position,
                                 relative_position_f,
                                 abs,
                                 zero,
                                 greater,
                                 less,
                                 condition,
                                 select_1,
                                 divide_1,
                                 log,
                                 divide_2,
                                 multiply_1,
                                 ceiling,
                                 add,
                                 multiply_2,
                                 select_2,
                                 two});
        mark_as_precision_sensitive(bucket->input(0));

        model_ref = make_shared<Model>(OutputVector{bucket}, ParameterVector{relative_position});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_keep_precission_sensitive_integer_to_float_subgraph) {
    // The marking must also survive the real f16 compression pass: the whole
    // integer -> float -> integer chain stays in f32.
    auto relative_position = make_shared<Parameter>(element::i64, Shape{2, 4});
    auto relative_position_f = make_shared<Convert>(relative_position, element::f32);
    auto abs = make_shared<Abs>(relative_position_f);
    auto ceiling = make_shared<Ceiling>(abs);
    auto bucket = make_shared<Convert>(ceiling, element::i64);
    auto model = make_shared<Model>(OutputVector{bucket}, ParameterVector{relative_position});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    EXPECT_EQ(relative_position_f->get_output_element_type(0), element::f32);
    EXPECT_EQ(abs->get_output_element_type(0), element::f32);
    EXPECT_EQ(ceiling->get_output_element_type(0), element::f32);
    EXPECT_EQ(bucket->get_input_element_type(0), element::f32);
    EXPECT_EQ(bucket->get_output_element_type(0), element::i64);
}

TEST(TransformationTests, do_not_mark_integer_to_float_subgraph_with_matmul) {
    // MatMul is a natural boundary of the elementwise index math, so the subgraph must not be marked.
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    auto relative_position = make_shared<Parameter>(element::i64, Shape{2, 4});
    auto relative_position_f = make_shared<Convert>(relative_position, element::f32);
    auto weights = make_shared<Parameter>(element::f32, Shape{4, 4});
    auto matmul = make_shared<MatMul>(relative_position_f, weights);
    auto bucket = make_shared<Convert>(matmul, element::i64);

    model = make_shared<Model>(OutputVector{bucket}, ParameterVector{relative_position, weights});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    EXPECT_FALSE(is_precision_sensitive(bucket->input(0)))
        << "a MatMul boundary must not make the integer result precision sensitive";
    EXPECT_FALSE(is_conversion_disabled(relative_position_f, element::f16));

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, do_not_mark_integer_to_float_subgraph_without_integer_result) {
    // The same integer-to-float conversion is not precision sensitive when its result stays floating point.
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;

    auto relative_position = make_shared<Parameter>(element::i64, Shape{2, 4});
    auto relative_position_f = make_shared<Convert>(relative_position, element::f32);
    auto abs = make_shared<Abs>(relative_position_f);
    auto sqrt = make_shared<Sqrt>(abs);

    model = make_shared<Model>(OutputVector{sqrt}, ParameterVector{relative_position});
    model_ref = model->clone();

    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    EXPECT_FALSE(is_conversion_disabled(relative_position_f, element::f16))
        << "a purely floating point result must stay compressible";

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
    result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, ConvertPrecision_preserves_integer_bucket_value) {
    // Numerical counterpart of ConvertPrecision_keep_precission_sensitive_integer_to_float_subgraph.
    // 48 * 1.0000625 is 48.003 in f32, so Ceiling must return 49. Near 48 the f16 spacing is
    // 0.03125, so a compressed chain rounds the value to 48.0 and Ceiling returns 48 instead.
    auto position = make_shared<Parameter>(element::i64, Shape{1});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto scale = Constant::create(element::f32, Shape{}, {1.0000625f});
    auto scaled = make_shared<Multiply>(position_f, scale);
    auto ceiling = make_shared<Ceiling>(scaled);
    auto bucket = make_shared<Convert>(ceiling, element::i64);
    auto model = make_shared<Model>(OutputVector{bucket}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    std::vector<int64_t> position_data{48};
    TensorVector inputs{Tensor(element::i64, Shape{1}, position_data.data())};
    TensorVector outputs(model->get_output_size());
    ASSERT_TRUE(model->evaluate(outputs, inputs));

    EXPECT_EQ(outputs[0].data<int64_t>()[0], 49)
        << "f16 compression must not change the integer bucket produced by Ceiling";
}

TEST(TransformationTests, ConvertPrecision_keeps_integer_rounding_ops_in_fp32) {
    // Ceiling, Floor and Round are all precision sensitive when they feed an integer conversion:
    // rounding the input to f16 first can move a value across the integer boundary, so every one
    // of them must keep the whole chain in f32. Each case picks an offset that f16 rounds across
    // the integer boundary, so the structural and the numerical check fail together without the
    // pass.
    struct RoundingCase {
        std::function<std::shared_ptr<Node>(const std::shared_ptr<Node>&)> make_rounding;
        float offset;
        int64_t expected;
    };
    const std::vector<RoundingCase> cases{{[](const std::shared_ptr<Node>& input) {
                                               return std::make_shared<Ceiling>(input);
                                           },
                                           0.003f,
                                           49},
                                          {[](const std::shared_ptr<Node>& input) {
                                               return std::make_shared<Floor>(input);
                                           },
                                           0.99999f,
                                           48},
                                          {[](const std::shared_ptr<Node>& input) {
                                               return std::make_shared<Round>(input,
                                                                              op::v5::Round::RoundMode::HALF_TO_EVEN);
                                           },
                                           0.50001f,
                                           49}};

    for (const auto& test_case : cases) {
        auto position = make_shared<Parameter>(element::i64, Shape{1});
        auto position_f = make_shared<Convert>(position, element::f32);
        auto offset = Constant::create(element::f32, Shape{}, {test_case.offset});
        auto shifted = make_shared<Add>(position_f, offset);
        auto rounded = test_case.make_rounding(shifted);
        auto bucket = make_shared<Convert>(rounded, element::i64);
        auto model = make_shared<Model>(OutputVector{bucket}, ParameterVector{position});

        pass::Manager manager;
        manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                      type_to_fuse_map{},
                                                      true);
        manager.run_passes(model);

        EXPECT_EQ(position_f->get_output_element_type(0), element::f32) << rounded->get_type_name();
        EXPECT_EQ(shifted->get_output_element_type(0), element::f32) << rounded->get_type_name();
        EXPECT_EQ(rounded->get_output_element_type(0), element::f32) << rounded->get_type_name();
        EXPECT_EQ(bucket->get_input_element_type(0), element::f32) << rounded->get_type_name();
        EXPECT_EQ(bucket->get_output_element_type(0), element::i64) << rounded->get_type_name();

        std::vector<int64_t> position_data{48};
        TensorVector inputs{Tensor(element::i64, Shape{1}, position_data.data())};
        TensorVector outputs(model->get_output_size());
        ASSERT_TRUE(model->evaluate(outputs, inputs));
        EXPECT_EQ(outputs[0].data<int64_t>()[0], test_case.expected) << rounded->get_type_name();
    }
}

TEST(TransformationTests, ConvertPrecision_preserves_truncating_conversion_without_rounding_op) {
    // Ceiling, Floor and Round are not the only way a rounding error becomes a discrete
    // difference: Convert(f32 -> i64) truncates, so a value that f32 places just below an
    // integer and f16 rounds up to it also changes the result. 47 * 0.9999998 is 46.99999 in
    // f32 and truncates to 46; in f16 the factor rounds to 1.0 and the result is 47.
    auto position = make_shared<Parameter>(element::i64, Shape{2});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto scale = Constant::create(element::f32, Shape{}, {0.9999998f});
    auto scaled = make_shared<Multiply>(position_f, scale);
    auto bucket = make_shared<Convert>(scaled, element::i64);
    auto model = make_shared<Model>(OutputVector{bucket}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    EXPECT_EQ(position_f->get_output_element_type(0), element::f32);
    EXPECT_EQ(scaled->get_output_element_type(0), element::f32);
    EXPECT_EQ(bucket->get_input_element_type(0), element::f32);

    std::vector<int64_t> position_data{47, -47};
    TensorVector inputs{Tensor(element::i64, Shape{2}, position_data.data())};
    TensorVector outputs(model->get_output_size());
    ASSERT_TRUE(model->evaluate(outputs, inputs));
    EXPECT_EQ(outputs[0].data<int64_t>()[0], 46);
    EXPECT_EQ(outputs[0].data<int64_t>()[1], -46);
}

TEST(TransformationTests, ConvertPrecision_keeps_unrelated_float_subgraph_in_fp16) {
    // The integer bucket chain must stay in f32, but an independent floating point branch that
    // never converts back to an integer must still be compressed.
    auto position = make_shared<Parameter>(element::i64, Shape{1});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto scale = Constant::create(element::f32, Shape{}, {1.0000625f});
    auto scaled = make_shared<Multiply>(position_f, scale);
    auto ceiling = make_shared<Ceiling>(scaled);
    auto bucket = make_shared<Convert>(ceiling, element::i64);

    auto data = make_shared<Parameter>(element::f32, Shape{1});
    auto factor = Constant::create(element::f32, Shape{}, {2.0f});
    auto unrelated = make_shared<Multiply>(data, factor);

    auto model = make_shared<Model>(OutputVector{bucket, unrelated}, ParameterVector{position, data});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    EXPECT_EQ(position_f->get_output_element_type(0), element::f32);
    EXPECT_EQ(scaled->get_output_element_type(0), element::f32);
    EXPECT_EQ(ceiling->get_output_element_type(0), element::f32);

    EXPECT_EQ(data->get_output_element_type(0), element::f16) << "unrelated parameter not compressed";
    EXPECT_EQ(unrelated->get_output_element_type(0), element::f16) << "unrelated branch not compressed";
}

TEST(TransformationTests, ConvertPrecision_keeps_forward_side_branch_in_fp16) {
    // Marking the source Convert would propagate down to every consumer, including a floating
    // point side branch that never reaches the integer conversion. Only the backward chain to the
    // terminal conversion is precision sensitive, so the side branch must still be compressed.
    auto position = make_shared<Parameter>(element::i64, Shape{1});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto scale = Constant::create(element::f32, Shape{}, {1.0000625f});
    auto scaled = make_shared<Multiply>(position_f, scale);
    auto ceiling = make_shared<Ceiling>(scaled);
    auto bucket = make_shared<Convert>(ceiling, element::i64);

    auto factor = Constant::create(element::f32, Shape{}, {2.0f});
    auto side_branch = make_shared<Multiply>(scaled, factor);

    auto model = make_shared<Model>(OutputVector{bucket, side_branch}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    EXPECT_EQ(position_f->get_output_element_type(0), element::f32);
    EXPECT_EQ(scaled->get_output_element_type(0), element::f32);
    EXPECT_EQ(ceiling->get_output_element_type(0), element::f32);
    EXPECT_EQ(side_branch->get_output_element_type(0), element::f16)
        << "a forward side branch of the protected chain must still be compressed";
}

TEST(TransformationTests, ConvertPrecision_compresses_integer_to_float_without_integer_terminal) {
    // An integer-to-float conversion whose result stays floating point is not an index
    // computation, so it must still be compressed to f16.
    auto position = make_shared<Parameter>(element::i64, Shape{1});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto abs = make_shared<Abs>(position_f);
    auto sqrt = make_shared<Sqrt>(abs);
    auto model = make_shared<Model>(OutputVector{sqrt}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    EXPECT_EQ(position_f->get_output_element_type(0), element::f16);
    EXPECT_EQ(abs->get_output_element_type(0), element::f16);
    EXPECT_EQ(sqrt->get_output_element_type(0), element::f16);
}

TEST(TransformationTests, ConvertPrecision_preserves_integer_bucket_selected_by_comparison) {
    // The int-to-float Convert here feeds only a boolean comparison. f16 rounds the
    // threshold 47.999 to 48.0, flipping Greater(48, threshold) from true to false and
    // selecting the wrong bucket. The boolean output must not terminate the subgraph walk.
    auto position = make_shared<Parameter>(element::i64, Shape{1});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto threshold = Constant::create(element::f32, Shape{}, {47.999f});
    auto cond = make_shared<Greater>(position_f, threshold);
    auto true_bucket = Constant::create(element::f32, Shape{}, {100.0f});
    auto false_bucket = Constant::create(element::f32, Shape{}, {200.0f});
    auto selected = make_shared<Select>(cond, true_bucket, false_bucket);
    auto bucket = make_shared<Convert>(selected, element::i64);
    auto model = make_shared<Model>(OutputVector{bucket}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    std::vector<int64_t> position_data{48};
    TensorVector inputs{Tensor(element::i64, Shape{1}, position_data.data())};
    TensorVector outputs(model->get_output_size());
    ASSERT_TRUE(model->evaluate(outputs, inputs));

    EXPECT_EQ(outputs[0].data<int64_t>()[0], 100)
        << "f16 compression must not flip a comparison that selects an integer bucket";
}

TEST(TransformationTests, ConvertPrecision_preserves_bucket_selected_by_boolean_chain) {
    // The sensitive value reaches Select only through the boolean condition:
    // Greater -> LogicalAnd -> Select(const, const). f16 rounds 47.999 to 48.0, so
    // Greater(48, 47.999) flips from true to false and the wrong constant is selected.
    auto position = make_shared<Parameter>(element::i64, Shape{1});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto lower = Constant::create(element::f32, Shape{}, {47.999f});
    auto upper = Constant::create(element::f32, Shape{}, {48.5f});
    auto above_lower = make_shared<Greater>(position_f, lower);
    auto below_upper = make_shared<Less>(position_f, upper);
    auto both = make_shared<LogicalAnd>(above_lower, below_upper);
    auto true_bucket = Constant::create(element::f32, Shape{}, {100.0f});
    auto false_bucket = Constant::create(element::f32, Shape{}, {200.0f});
    auto selected = make_shared<Select>(both, true_bucket, false_bucket);
    auto bucket = make_shared<Convert>(selected, element::i64);
    auto model = make_shared<Model>(OutputVector{bucket}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    std::vector<int64_t> position_data{48};
    TensorVector inputs{Tensor(element::i64, Shape{1}, position_data.data())};
    TensorVector outputs(model->get_output_size());
    ASSERT_TRUE(model->evaluate(outputs, inputs));

    EXPECT_EQ(outputs[0].data<int64_t>()[0], 100)
        << "f16 compression must not flip a boolean chain that selects an integer bucket";
}

TEST(TransformationTests, ConvertPrecision_preserves_boolean_terminal_conversion) {
    // The float chain ends in Convert(boolean -> i64) rather than Convert(f32 -> i64).
    // The comparison result still has to be computed from f32 values.
    auto position = make_shared<Parameter>(element::i64, Shape{1});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto threshold = Constant::create(element::f32, Shape{}, {47.999f});
    auto cond = make_shared<Greater>(position_f, threshold);
    auto bucket = make_shared<Convert>(cond, element::i64);
    auto model = make_shared<Model>(OutputVector{bucket}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::ConvertPrecision>(precisions_map{{element::f32, element::f16}},
                                                  type_to_fuse_map{},
                                                  true);
    manager.run_passes(model);

    std::vector<int64_t> position_data{48};
    TensorVector inputs{Tensor(element::i64, Shape{1}, position_data.data())};
    TensorVector outputs(model->get_output_size());
    ASSERT_TRUE(model->evaluate(outputs, inputs));

    EXPECT_EQ(outputs[0].data<int64_t>()[0], 1) << "f16 compression must not flip a comparison converted to an integer";
}

TEST(TransformationTests, MarkIntegerToFloatSubgraph_follows_all_split_outputs) {
    // Split has several outputs; the walk must continue from every one of them.
    auto position = make_shared<Parameter>(element::i64, Shape{2, 4});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto axis = Constant::create(element::i64, Shape{}, {1});
    auto split = make_shared<Split>(position_f, axis, 2);
    auto ceiling1 = make_shared<Ceiling>(split->output(0));
    auto bucket1 = make_shared<Convert>(ceiling1, element::i64);
    auto ceiling2 = make_shared<Ceiling>(split->output(1));
    auto bucket2 = make_shared<Convert>(ceiling2, element::i64);
    auto model = make_shared<Model>(OutputVector{bucket1, bucket2}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    EXPECT_TRUE(is_precision_sensitive(bucket1->input(0))) << "first Split output not followed";
    EXPECT_TRUE(is_precision_sensitive(bucket2->input(0))) << "second Split output not followed";
}

TEST(TransformationTests, MarkIntegerToFloatSubgraph_marks_every_terminal_convert) {
    // One int-to-float Convert can feed more than one float-to-int Convert through different
    // branches. Every terminal conversion needs its input marked, not just the first one found.
    auto position = make_shared<Parameter>(element::i64, Shape{1});
    auto position_f = make_shared<Convert>(position, element::f32);
    auto scale = Constant::create(element::f32, Shape{}, {1.0000625f});
    auto scaled = make_shared<Multiply>(position_f, scale);
    auto ceiling1 = make_shared<Ceiling>(scaled);
    auto bucket1 = make_shared<Convert>(ceiling1, element::i64);
    auto offset = Constant::create(element::f32, Shape{}, {0.5f});
    auto shifted = make_shared<Add>(scaled, offset);
    auto ceiling2 = make_shared<Ceiling>(shifted);
    auto bucket2 = make_shared<Convert>(ceiling2, element::i64);
    auto model = make_shared<Model>(OutputVector{bucket1, bucket2}, ParameterVector{position});

    pass::Manager manager;
    manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
    manager.run_passes(model);

    EXPECT_TRUE(is_precision_sensitive(bucket1->input(0))) << "first terminal convert not marked";
    EXPECT_TRUE(is_precision_sensitive(bucket2->input(0))) << "second terminal convert not marked";
}