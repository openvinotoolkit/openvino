// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/sdpa_scale_fusion.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "ov_ops/type_relaxed.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov;

TEST_F(TransformationTestsF, SDPAScaleFusionTest1) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{8.0f});
    const auto v_scaled = std::make_shared<ov::op::v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale_const);
        const auto k_scaled = std::make_shared<ov::op::v1::Multiply>(key, scale_const);
        const auto sdpa =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled, k_scaled, v_scaled, casual);

        model = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto new_mask_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
        const auto new_scale_const =
            ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{64.0f / std::sqrt(32.0f)});
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   v_scaled,
                                                                                   new_mask_const,
                                                                                   new_scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest2) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto sdpa_mask_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{2.0f});
    const auto scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{8.0f});
    const auto v_scaled = std::make_shared<ov::op::v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale_const);
        const auto k_scaled = std::make_shared<ov::op::v1::Multiply>(key, scale_const);
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled,
                                                                                   k_scaled,
                                                                                   v_scaled,
                                                                                   sdpa_mask_const,
                                                                                   sdpa_scale_const,
                                                                                   casual);

        model = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto new_scale_const =
            ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{128.0f});
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   v_scaled,
                                                                                   sdpa_mask_const,
                                                                                   new_scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest3) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto sdpa_mask_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{2.0f});
    const auto scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{8.0f});
    const auto v_scaled = std::make_shared<ov::op::v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale_const);
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled,
                                                                                   key,
                                                                                   v_scaled,
                                                                                   sdpa_mask_const,
                                                                                   sdpa_scale_const,
                                                                                   casual);

        model = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto new_scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{16.0f});
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   v_scaled,
                                                                                   sdpa_mask_const,
                                                                                   new_scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest4) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto sdpa_mask_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{2.0f});
    const auto scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{8.0f});
    const auto scale_dyn = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::Shape{});
    const auto v_scaled = std::make_shared<ov::op::v1::Multiply>(value, scale_const);
    const auto casual = false;
    const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale_dyn);
    {
        const auto k_scaled = std::make_shared<ov::op::v1::Multiply>(key, scale_const);
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled,
                                                                                   k_scaled,
                                                                                   v_scaled,
                                                                                   sdpa_mask_const,
                                                                                   sdpa_scale_const,
                                                                                   casual);

        model = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value, scale_dyn});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto new_scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{16.0f});
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled,
                                                                                   key,
                                                                                   v_scaled,
                                                                                   sdpa_mask_const,
                                                                                   new_scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value, scale_dyn});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest5) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    const auto sdpa_mask_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{1.0f});
    const auto scale_const = ov::op::v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{1.0f});
    const auto scale_dyn = std::make_shared<ov::op::v0::Parameter>(element::f32, ov::Shape{});
    const auto v_scaled = std::make_shared<ov::op::v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale_dyn);
        const auto k_scaled = std::make_shared<ov::op::v1::Multiply>(key, scale_const);
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled,
                                                                                   k_scaled,
                                                                                   v_scaled,
                                                                                   sdpa_mask_const,
                                                                                   sdpa_scale_const,
                                                                                   casual);

        model = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value, scale_dyn});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   key,
                                                                                   v_scaled,
                                                                                   sdpa_mask_const,
                                                                                   scale_dyn,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value, scale_dyn});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest6) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::i8, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f16, value_shape);
    const auto scale_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{8.0f});
    const auto v_scaled = std::make_shared<ov::op::v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale_const);
        const auto k_scaled = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{element::f16, element::f16},
            std::vector<element::Type>{element::f16},
            ov::op::TemporaryReplaceOutputType(key, element::f16).get(),
            ov::op::TemporaryReplaceOutputType(scale_const, element::f16).get());
        const auto sdpa =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled, k_scaled, v_scaled, casual);

        model = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto k_scaled_ref = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{element::f16, element::f16},
            std::vector<element::Type>{element::f16},
            ov::op::TemporaryReplaceOutputType(key, element::f16).get(),
            ov::op::TemporaryReplaceOutputType(scale_const, element::f16).get());
        const auto new_mask_const = ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{0.0f});
        const auto new_scale_const =
            ov::op::v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{8.0f / std::sqrt(32.0f)});
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                                   k_scaled_ref,
                                                                                   v_scaled,
                                                                                   new_mask_const,
                                                                                   new_scale_const,
                                                                                   casual);
        model_ref = std::make_shared<ov::Model>(NodeVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
