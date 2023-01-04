// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/align_mixed_fp32_fp16_types.hpp>
#include <transformations/common_optimizations/mark_subgraphs_to_keep_in_mixed_precision.hpp>
#include <transformations/convert_precision.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;
using namespace std;

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_1) {
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

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        type_to_fuse_map empty_type_to_fuse_map = {};
        manager.register_pass<pass::ConvertPrecision>(convert_precision_list, empty_type_to_fuse_map, true);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto convert_to_f32_1 = make_shared<opset8::Convert>(input_1, element::f32);
        auto exp_1 = make_shared<opset8::Exp>(convert_to_f32_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto convert_to_f16_1 = make_shared<opset8::Convert>(mul_1, element::f16);
        auto matmul_1 = make_shared<opset8::MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_2) {
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

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        type_to_fuse_map empty_type_to_fuse_map = {};
        manager.register_pass<pass::ConvertPrecision>(convert_precision_list, empty_type_to_fuse_map, true);

        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto convert_to_f32_1 = make_shared<opset8::Convert>(input_1, element::f32);
        auto exp_1 = make_shared<opset8::Exp>(convert_to_f32_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<opset8::Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, factor_const_decompressed);
        auto convert_to_f16_1 = make_shared<opset8::Convert>(mul_1, element::f16);
        auto matmul_1 = make_shared<opset8::MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_3) {
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

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        type_to_fuse_map empty_type_to_fuse_map = {};
        manager.register_pass<pass::ConvertPrecision>(convert_precision_list, empty_type_to_fuse_map, true);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto convert_to_f32_1 = make_shared<opset8::Convert>(input_1, element::f32);
        auto exp_1 = make_shared<opset8::Exp>(convert_to_f32_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});

        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Add>(reduce_sum_1, addition_const);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<opset8::Multiply>(add_1, factor_const_decompressed);
        auto convert_to_f16_1 = make_shared<opset8::Convert>(mul_1, element::f16);
        auto matmul_1 = make_shared<opset8::MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_4) {
    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto mvn_1 = make_shared<opset8::MVN>(input_1, reduction_axes, true, 1.e-8, op::MVNEpsMode::INSIDE_SQRT);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Add>(mvn_1, addition_const);
        auto matmul_1 = make_shared<opset8::MatMul>(add_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        type_to_fuse_map empty_type_to_fuse_map = {};
        manager.register_pass<pass::ConvertPrecision>(convert_precision_list, empty_type_to_fuse_map, true);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto convert_to_f32_1 = make_shared<opset8::Convert>(input_1, element::f32);
        auto mvn_1 =
            make_shared<opset8::MVN>(convert_to_f32_1, reduction_axes, true, 1.e-8, op::MVNEpsMode::INSIDE_SQRT);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Add>(mvn_1, addition_const);
        auto convert_to_f16_1 = make_shared<opset8::Convert>(add_1, element::f16);
        auto matmul_1 = make_shared<opset8::MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_mnv_with_split) {
    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 56, 224});

        auto split_axis = opset8::Constant::create(element::i64, Shape{}, {3});
        auto split = make_shared<opset8::Split>(input_1, split_axis, 4);

        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto mvn_1 =
            make_shared<opset8::MVN>(split->output(0), reduction_axes, true, 1.e-8, op::MVNEpsMode::INSIDE_SQRT);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Add>(mvn_1, addition_const);
        auto matmul_1 = make_shared<opset8::MatMul>(add_1, input_2);

        auto result_1 = make_shared<opset8::Result>(matmul_1);
        auto result_2 = make_shared<opset8::Result>(split->output(1));
        auto result_3 = make_shared<opset8::Result>(split->output(2));
        model = make_shared<Model>(OutputVector{result_1, result_2, result_3}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        type_to_fuse_map empty_type_to_fuse_map = {};
        manager.register_pass<pass::ConvertPrecision>(convert_precision_list, empty_type_to_fuse_map, true);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto input_2 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 56, 224});

        auto convert_to_f32_1 = make_shared<opset8::Convert>(input_1, element::f32);

        auto split_axis = opset8::Constant::create(element::i64, Shape{}, {3});
        auto split = make_shared<opset8::Split>(convert_to_f32_1, split_axis, 4);

        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto mvn_1 =
            make_shared<opset8::MVN>(split->output(0), reduction_axes, true, 1.e-8, op::MVNEpsMode::INSIDE_SQRT);
        auto addition_const = opset8::Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<opset8::Add>(mvn_1, addition_const);
        auto convert_to_f16_1 = make_shared<opset8::Convert>(add_1, element::f16);
        auto matmul_1 = make_shared<opset8::MatMul>(convert_to_f16_1, input_2);

        // todo: without Converts to fp16 because of GPU
        auto result_1 = make_shared<opset8::Result>(matmul_1);
        auto result_2 = make_shared<opset8::Result>(split->output(1));
        auto result_3 = make_shared<opset8::Result>(split->output(2));

        model_ref = make_shared<Model>(OutputVector{result_1, result_2, result_3}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_with_rand_uniform) {
    {
        auto input_1 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<opset8::Exp>(input_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);

        auto out_shape = opset8::Constant::create(element::i64, Shape{3}, {1, 3, 224});
        auto minval = opset8::Constant::create(element::f32, Shape{}, {1});
        auto maxval = opset8::Constant::create(element::f32, Shape{}, {10});
        auto rand_uniform = make_shared<opset8::RandomUniform>(out_shape, minval, maxval, element::f32);
        auto rand_uniform_add_factor = make_shared<opset8::Add>(rand_uniform, factor_const_decompressed);

        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, rand_uniform_add_factor);
        auto matmul_1 = make_shared<opset8::MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        type_to_fuse_map empty_type_to_fuse_map = {};
        manager.register_pass<pass::ConvertPrecision>(convert_precision_list, empty_type_to_fuse_map, true);
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto convert_to_f32_1 = make_shared<opset8::Convert>(input_1, element::f32);
        auto exp_1 = make_shared<opset8::Exp>(convert_to_f32_1);
        auto input_2 = make_shared<opset8::Parameter>(element::f16, Shape{1, 3, 224, 224});
        auto reduction_axes = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<opset8::ReduceSum>(exp_1, reduction_axes);

        auto factor_const = opset8::Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<opset8::Convert>(factor_const, element::f32);

        auto out_shape = opset8::Constant::create(element::i64, Shape{3}, {1, 3, 224});
        auto minval = opset8::Constant::create(element::f16, Shape{}, {1});
        auto maxval = opset8::Constant::create(element::f16, Shape{}, {10});
        auto rand_uniform = make_shared<opset8::RandomUniform>(out_shape, minval, maxval, element::f16);
        auto rand_uniform_decompressed = make_shared<opset8::Convert>(rand_uniform, element::f32);
        auto rand_uniform_add_factor = make_shared<opset8::Add>(rand_uniform_decompressed, factor_const_decompressed);

        auto mul_1 = make_shared<opset8::Multiply>(reduce_sum_1, rand_uniform_add_factor);
        auto convert_to_f16_1 = make_shared<opset8::Convert>(mul_1, element::f16);
        auto matmul_1 = make_shared<opset8::MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}
