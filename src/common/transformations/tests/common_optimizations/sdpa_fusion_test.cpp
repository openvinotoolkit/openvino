// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/sdpa_fusion.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov;
using namespace ov::op;

TEST_F(TransformationTestsF, SDPAFusionTest1) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto casual = false;
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key, false, true);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{1.0f});
        const auto mask_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   value,
                                                                                   mask_const,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest2) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto casual = false;
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key, false, true);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{1.0f});
        const auto mask_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{0.0f});
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   value,
                                                                                   mask_const,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest3) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto casual = false;

    const auto key_t =
        std::make_shared<ov::op::v1::Transpose>(key,
                                                ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key_t, false, false);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{1.0f});
        const auto mask_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{0.0f});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
        auto backward_transpose = std::make_shared<ov::op::v1::Transpose>(key_t, axes);
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   backward_transpose,
                                                                                   value,
                                                                                   mask_const,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest4) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, 32, -1};
    const PartialShape value_shape{1, 32, -1, 32};
    const auto casual = false;

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key, false, false);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{1.0f});
        const auto mask_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{0.0f});
        const auto transposed_key =
            std::make_shared<ov::op::v1::Transpose>(key,
                                                    ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   transposed_key,
                                                                                   value,
                                                                                   mask_const,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest5) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};
    const PartialShape attention_mask_shape{1, 32, -1, -1};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto mask = std::make_shared<ov::op::v0::Parameter>(element::f16, attention_mask_shape);
    const auto casual = false;
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key, false, true);
        const auto mask_add = std::make_shared<ov::op::v1::Add>(qk, mask);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(mask_add, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{1.0f});
        const auto squeezed_mask =
            std::make_shared<ov::op::v0::Squeeze>(mask, ov::op::v0::Constant::create(element::i64, Shape{1}, {0}));
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   value,
                                                                                   squeezed_mask,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, mask});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest6) {
    const PartialShape query_shape{1, 32, 10, 32};
    const PartialShape key_shape{1, 32, 10, 32};
    const PartialShape value_shape{1, 32, 10, 32};
    const PartialShape attention_mask_shape{1, 1, 10, 10};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto mask = std::make_shared<ov::op::v0::Parameter>(element::f16, attention_mask_shape);
    const auto casual = false;
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key, false, true);
        const auto mask_add = std::make_shared<ov::op::v1::Add>(qk, mask);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(mask_add, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{1.0f});
        const auto squeezed_mask =
            std::make_shared<ov::op::v0::Squeeze>(mask, ov::op::v0::Constant::create(element::i64, Shape{2}, {0, 1}));
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   value,
                                                                                   squeezed_mask,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, mask});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest7) {
    const PartialShape query_shape{1, 8, -1, 32};
    const PartialShape key_shape{-1, 1, 8, 32};
    const PartialShape value_shape{1, 8, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto key_t =
        std::make_shared<ov::op::v1::Transpose>(key,
                                                ov::op::v0::Constant::create(element::i64, Shape{4}, {1, 2, 3, 0}));

    const auto mask = ov::op::v0::Constant::create(element::f16, ov::Shape{}, {0});
    const auto scale = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
    const auto casual = false;
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key_t, false, false);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
        auto backward_transpose = std::make_shared<ov::op::v1::Transpose>(key_t, axes);
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   backward_transpose,
                                                                                   value,
                                                                                   mask,
                                                                                   scale,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest8) {
    const PartialShape query_shape{1, 10, 1024, 64};
    const PartialShape key_shape{1, 10, 1024, 64};
    const PartialShape value_shape{1, 10, 1024, 64};
    const PartialShape attention_mask_shape{10, 1024, 1024};
    const Shape scale_shape{1};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto mask = std::make_shared<ov::op::v0::Parameter>(element::f16, attention_mask_shape);
    const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
    const auto casual = false;
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key, false, true);
        const auto mask_add = std::make_shared<ov::op::v1::Add>(qk, mask);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(mask_add, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto sdpa =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, scale_const, casual);

        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, mask});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest9) {
    const PartialShape query_shape{1, 10, 1024, 64};
    const PartialShape key_shape{1, 10, 1024, 64};
    const PartialShape value_shape{1, 10, 1024, 64};
    const PartialShape attention_mask_shape{10, 1024, 1024};
    const Shape scale_shape{1};

    const PartialShape query_reshaped_shape{10, 1024, 64};
    const PartialShape key_reshaped_shape{10, 1024, 64};
    const PartialShape value_reshaped_shape{10, 1024, 64};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto mask = std::make_shared<ov::op::v0::Parameter>(element::f16, attention_mask_shape);
    const auto casual = false;
    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
        const auto query_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                       Shape{query_reshaped_shape.size()},
                                                                       query_reshaped_shape.to_shape());
        const auto query_reshaped = std::make_shared<ov::op::v1::Reshape>(query, query_reshape_params, true);

        const auto key_reshape_params =
            ov::op::v0::Constant::create(element::i64, Shape{key_reshaped_shape.size()}, key_reshaped_shape.to_shape());
        const auto key_reshaped = std::make_shared<ov::op::v1::Reshape>(key, key_reshape_params, true);

        const auto value_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                       Shape{value_reshaped_shape.size()},
                                                                       value_reshaped_shape.to_shape());
        const auto value_reshaped = std::make_shared<ov::op::v1::Reshape>(value, value_reshape_params, true);

        const auto transposed_key = std::make_shared<ov::op::v1::Transpose>(
            key_reshaped,
            ov::op::v0::Constant::create(element::i64, Shape{key_reshaped_shape.size()}, {0, 2, 1}));

        const auto qk = std::make_shared<ov::op::v0::MatMul>(query_reshaped, transposed_key, false, false);
        const auto scaled_qk = std::make_shared<ov::op::v1::Multiply>(qk, scale_const);
        const auto mask_add = std::make_shared<ov::op::v1::Add>(scaled_qk, mask);

        const auto softmax = std::make_shared<ov::op::v8::Softmax>(mask_add, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value_reshaped, false, false);

        const auto reshape_result =
            ov::op::v0::Constant::create(element::i64, ov::Shape{query_shape.size()}, query_shape.to_shape());
        const auto qkv_result = std::make_shared<ov::op::v1::Reshape>(qkv, reshape_result, true);

        model = std::make_shared<ov::Model>(OutputVector{qkv_result}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
        const auto sdpa =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, scale_const, casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, mask});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest10) {
    const PartialShape query_shape{1, 10, 1024, 64};
    const PartialShape key_shape{1, 10, 77, 64};
    const PartialShape value_shape{1, 10, 77, 64};
    const Shape scale_shape{1};

    const PartialShape query_reshaped_shape{10, 1024, 64};
    const PartialShape key_reshaped_shape{10, 77, 64};
    const PartialShape value_reshaped_shape{10, 77, 64};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto casual = false;
    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
        const auto query_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                       Shape{query_reshaped_shape.size()},
                                                                       query_reshaped_shape.to_shape());
        const auto query_reshaped = std::make_shared<ov::op::v1::Reshape>(query, query_reshape_params, true);

        const auto key_reshape_params =
            ov::op::v0::Constant::create(element::i64, Shape{key_reshaped_shape.size()}, key_reshaped_shape.to_shape());
        const auto key_reshaped = std::make_shared<ov::op::v1::Reshape>(key, key_reshape_params, true);

        const auto value_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                       Shape{value_reshaped_shape.size()},
                                                                       value_reshaped_shape.to_shape());
        const auto value_reshaped = std::make_shared<ov::op::v1::Reshape>(value, value_reshape_params, true);

        const auto scaled_key = std::make_shared<ov::op::v1::Multiply>(key_reshaped, scale_const);

        const auto qk = std::make_shared<ov::op::v0::MatMul>(query_reshaped, scaled_key, false, true);

        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);

        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value_reshaped, false, false);
        const auto reshape_result =
            ov::op::v0::Constant::create(element::i64, ov::Shape{query_shape.size()}, query_shape.to_shape());
        const auto qkv_result = std::make_shared<ov::op::v1::Reshape>(qkv, reshape_result, true);

        model = std::make_shared<ov::Model>(OutputVector{qkv_result}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
        const auto mask_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{0.0f});
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   value,
                                                                                   mask_const,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest11) {
    const PartialShape query_shape{1, 10, 1024, 64};
    const PartialShape key_shape{1, 10, 77, 64};
    const PartialShape value_shape{1, 10, 77, 64};
    const PartialShape attention_mask_shape{10, 1024, 77};
    const Shape scale_shape{1};

    const PartialShape query_reshaped_shape{10, 1024, 64};
    const PartialShape key_reshaped_shape{10, 77, 64};
    const PartialShape value_reshaped_shape{10, 77, 64};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto mask = std::make_shared<ov::op::v0::Parameter>(element::f16, attention_mask_shape);
    const auto casual = false;
    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
        const auto query_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                       Shape{query_reshaped_shape.size()},
                                                                       query_reshaped_shape.to_shape());
        const auto query_reshaped = std::make_shared<ov::op::v1::Reshape>(query, query_reshape_params, true);

        const auto key_reshape_params =
            ov::op::v0::Constant::create(element::i64, Shape{key_reshaped_shape.size()}, key_reshaped_shape.to_shape());
        const auto key_reshaped = std::make_shared<ov::op::v1::Reshape>(key, key_reshape_params, true);

        const auto value_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                       Shape{value_reshaped_shape.size()},
                                                                       value_reshaped_shape.to_shape());
        const auto value_reshaped = std::make_shared<ov::op::v1::Reshape>(value, value_reshape_params, true);

        const auto transposed_key = std::make_shared<ov::op::v1::Transpose>(
            key_reshaped,
            ov::op::v0::Constant::create(element::i64, Shape{key_reshaped_shape.size()}, {0, 2, 1}));

        const auto qk = std::make_shared<ov::op::v0::MatMul>(query_reshaped, transposed_key, false, false);
        const auto scaled_qk = std::make_shared<ov::op::v1::Multiply>(qk, scale_const);

        const auto mask_add = std::make_shared<ov::op::v1::Add>(scaled_qk, mask);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(mask_add, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value_reshaped, false, false);

        const auto reshape_result =
            ov::op::v0::Constant::create(element::i64, ov::Shape{query_shape.size()}, query_shape.to_shape());
        const auto qkv_result = std::make_shared<ov::op::v1::Reshape>(qkv, reshape_result, true);

        model = std::make_shared<ov::Model>(OutputVector{qkv_result}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
        const auto sdpa =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, scale_const, casual);
        const auto reshape_result = model_ref =
            std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, mask});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest12) {
    const PartialShape query_shape{1, 1, 49, 128};
    const PartialShape key_shape{1, 128, 1, 49};
    const PartialShape value_shape{1, 1, 49, 128};
    const PartialShape attention_mask_shape{1, 1, 49, 49};
    const Shape scale_shape{1};

    const PartialShape query_reshaped_shape{49, 128};
    const PartialShape key_reshaped_shape{128, 49};
    const PartialShape value_reshaped_shape{49, 128};
    const PartialShape attention_mask_reshaped_shape{1, 49, 49};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto mask = std::make_shared<ov::op::v0::Parameter>(element::f16, attention_mask_shape);
    const auto casual = false;
    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
        const auto query_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                       Shape{query_reshaped_shape.size()},
                                                                       query_reshaped_shape.to_shape());
        const auto query_reshaped = std::make_shared<ov::op::v1::Reshape>(query, query_reshape_params, true);

        const auto key_reshape_params =
            ov::op::v0::Constant::create(element::i64, Shape{key_reshaped_shape.size()}, key_reshaped_shape.to_shape());
        const auto key_reshaped = std::make_shared<ov::op::v1::Reshape>(key, key_reshape_params, true);

        const auto value_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                       Shape{value_reshaped_shape.size()},
                                                                       value_reshaped_shape.to_shape());
        const auto value_reshaped = std::make_shared<ov::op::v1::Reshape>(value, value_reshape_params, true);

        const auto qk = std::make_shared<ov::op::v0::MatMul>(query_reshaped, key_reshaped, false, false);
        const auto qk_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                    Shape{attention_mask_reshaped_shape.size()},
                                                                    attention_mask_reshaped_shape.to_shape());
        const auto qk_reshaped = std::make_shared<ov::op::v1::Reshape>(qk, qk_reshape_params, true);

        const auto scaled_qk = std::make_shared<ov::op::v1::Multiply>(qk_reshaped, scale_const);
        const auto qk_scaled_reshape_params = ov::op::v0::Constant::create(element::i64,
                                                                           Shape{attention_mask_shape.size()},
                                                                           attention_mask_shape.to_shape());
        const auto scaled_qk_reshaped =
            std::make_shared<ov::op::v1::Reshape>(scaled_qk, qk_scaled_reshape_params, true);

        const auto mask_add = std::make_shared<ov::op::v1::Add>(scaled_qk_reshaped, mask);

        const auto softmax = std::make_shared<ov::op::v8::Softmax>(mask_add, -1);
        const auto softmax_reshape_params = ov::op::v0::Constant::create(
            element::i64,
            Shape{2},
            {attention_mask_shape[-2].get_max_length(), attention_mask_shape[-1].get_max_length()});
        const auto softmax_reshaped = std::make_shared<ov::op::v1::Reshape>(softmax, softmax_reshape_params, true);

        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax_reshaped, value_reshaped, false, false);

        const auto reshape_result =
            ov::op::v0::Constant::create(element::i64, ov::Shape{query_shape.size()}, query_shape.to_shape());
        const auto qkv_result = std::make_shared<ov::op::v1::Reshape>(qkv, reshape_result, true);

        model = std::make_shared<ov::Model>(OutputVector{qkv_result}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, {}, std::vector<float>{1.0f});
        const auto transposed_key =
            std::make_shared<ov::op::v1::Transpose>(key,
                                                    ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1}));
        const auto squeezed_mask =
            std::make_shared<ov::op::v0::Squeeze>(mask, ov::op::v0::Constant::create(element::i64, Shape{2}, {0, 1}));
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   transposed_key,
                                                                                   value,
                                                                                   squeezed_mask,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, mask});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest_DynamicBatch) {
    const PartialShape query_shape{-1, 32, -1, 32};
    const PartialShape key_shape{-1, 32, -1, 32};
    const PartialShape value_shape{-1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto casual = false;

    const auto key_t =
        std::make_shared<ov::op::v1::Transpose>(key,
                                                ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key_t, false, false);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(OutputVector{qkv}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{1.0f});
        const auto mask_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{0.0f});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
        auto backward_transpose = std::make_shared<ov::op::v1::Transpose>(key_t, axes);
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   backward_transpose,
                                                                                   value,
                                                                                   mask_const,
                                                                                   scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest_4dAttentionMaskWithBatch2) {
    // Init.
    int64_t batch = 2;
    const PartialShape query_shape{batch, 49, 52};
    const PartialShape key_shape{batch, 49, 52};
    const PartialShape value_shape{batch, 49, 52};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);

    // Preprocessing callback.
    auto callback = [](auto& nodes) {
        int64_t axis = 0;
        auto axes_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        nodes[InputType::Q] = std::make_shared<v0::Unsqueeze>(nodes[InputType::Q], axes_node)->output(0);

        auto axes_node1 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        nodes[InputType::K] = std::make_shared<v0::Unsqueeze>(nodes[InputType::K], axes_node1)->output(0);

        auto axes_node2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        nodes[InputType::V] = std::make_shared<v0::Unsqueeze>(nodes[InputType::V], axes_node2)->output(0);
    };

    // SDPA model.
    {
        sdpa.set_mask({batch, 1, 1, 49});
        sdpa.create_pattern_sdpa(true);

        model = sdpa.build_model();

        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.set_mask({batch, 1, 1, 49});
        sdpa_ref.set_preprocessing_callback(callback);
        sdpa_ref.create_reference_sdpa();

        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}