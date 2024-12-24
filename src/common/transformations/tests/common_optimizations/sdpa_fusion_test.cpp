// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/sdpa_fusion.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov;

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

        model = std::make_shared<ov::Model>(NodeVector{qkv}, ParameterVector{query, key, value});
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
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
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

        model = std::make_shared<ov::Model>(NodeVector{qkv}, ParameterVector{query, key, value});
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
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
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
    {
        const auto key_t =
            std::make_shared<ov::op::v1::Transpose>(key,
                                                    op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key_t, false, false);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(NodeVector{qkv}, ParameterVector{query, key, value});
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
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest4) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, 32, -1};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    {
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key, false, false);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(NodeVector{qkv}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    model_ref = model->clone();

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

        model = std::make_shared<ov::Model>(NodeVector{qkv}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{1.0f});
        const auto sdpa =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, scale_const, casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value, mask});
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

        model = std::make_shared<ov::Model>(NodeVector{qkv}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    {
        const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{1.0f});
        const auto sdpa =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, scale_const, casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value, mask});
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
    {
        const auto key_t =
            std::make_shared<ov::op::v1::Transpose>(key,
                                                    op::v0::Constant::create(element::i64, Shape{4}, {1, 2, 3, 0}));
        const auto qk = std::make_shared<ov::op::v0::MatMul>(query, key_t, false, false);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(qk, -1);
        const auto qkv = std::make_shared<ov::op::v0::MatMul>(softmax, value, false, false);

        model = std::make_shared<ov::Model>(NodeVector{qkv}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAFusion>();
    }
}
