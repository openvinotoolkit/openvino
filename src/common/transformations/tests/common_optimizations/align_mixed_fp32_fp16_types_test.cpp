// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/align_mixed_fp32_fp16_types.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/fp16_compression/mark_subgraphs_to_keep_in_mixed_precision.hpp"

using namespace testing;
using namespace ov;
using namespace std;
using namespace ov::opset10;

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_1) {
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

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto convert_to_f32_1 = make_shared<Convert>(input_1, element::f32);
        auto exp_1 = make_shared<Exp>(convert_to_f32_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto convert_to_f16_1 = make_shared<Convert>(mul_1, element::f32);
        auto matmul_1 = make_shared<MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_2) {
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

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();

        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto convert_to_f32_1 = make_shared<Convert>(input_1, element::f32);
        auto exp_1 = make_shared<Exp>(convert_to_f32_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});

        auto unsqueeze_axes = Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_1 = make_shared<Unsqueeze>(exp_1, unsqueeze_axes);
        auto reduce_sum_1 = make_shared<ReduceSum>(unsqueeze_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(reduce_sum_1, factor_const_decompressed);
        auto convert_to_f16_1 = make_shared<Convert>(mul_1, element::f32);
        auto matmul_1 = make_shared<MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_3) {
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

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto convert_to_f32_1 = make_shared<Convert>(input_1, element::f32);
        auto exp_1 = make_shared<Exp>(convert_to_f32_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});

        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);
        auto addition_const = Constant::create(element::f32, Shape{1}, {0.1f});
        auto add_1 = make_shared<Add>(reduce_sum_1, addition_const);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);
        auto mul_1 = make_shared<Multiply>(add_1, factor_const_decompressed);
        auto convert_to_f16_1 = make_shared<Convert>(mul_1, element::f32);
        auto matmul_1 = make_shared<MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, align_mixed_fp16_fp32_with_rand_uniform) {
    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto exp_1 = make_shared<Exp>(input_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);

        auto out_shape = Constant::create(element::i64, Shape{3}, {1, 3, 224});
        auto minval = Constant::create(element::f32, Shape{}, {1});
        auto maxval = Constant::create(element::f32, Shape{}, {10});
        auto rand_uniform = make_shared<RandomUniform>(out_shape, minval, maxval, element::f32);
        auto rand_uniform_add_factor = make_shared<Add>(rand_uniform, factor_const_decompressed);

        auto mul_1 = make_shared<Multiply>(reduce_sum_1, rand_uniform_add_factor);
        auto matmul_1 = make_shared<MatMul>(mul_1, input_2);

        model = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        manager.run_passes(model);
    }

    {
        auto input_1 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto convert_to_f32_1 = make_shared<Convert>(input_1, element::f32);
        auto exp_1 = make_shared<Exp>(convert_to_f32_1);
        auto input_2 = make_shared<Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto reduction_axes = Constant::create(element::i64, Shape{1}, {-1});
        auto reduce_sum_1 = make_shared<ReduceSum>(exp_1, reduction_axes);

        auto factor_const = Constant::create(element::f16, Shape{1}, {-1});
        auto factor_const_decompressed = make_shared<Convert>(factor_const, element::f32);

        auto out_shape = Constant::create(element::i64, Shape{3}, {1, 3, 224});
        auto minval = Constant::create(element::f32, Shape{}, {1});
        auto maxval = Constant::create(element::f32, Shape{}, {10});
        auto rand_uniform = make_shared<RandomUniform>(out_shape, minval, maxval, element::f32);
        auto rand_uniform_decompressed = make_shared<Convert>(rand_uniform, element::f32);
        auto rand_uniform_add_factor = make_shared<Add>(rand_uniform_decompressed, factor_const_decompressed);

        auto mul_1 = make_shared<Multiply>(reduce_sum_1, rand_uniform_add_factor);
        auto convert_to_f16_1 = make_shared<Convert>(mul_1, element::f32);
        auto matmul_1 = make_shared<MatMul>(convert_to_f16_1, input_2);

        model_ref = make_shared<Model>(NodeVector{matmul_1}, ParameterVector{input_1, input_2});
    }
}
