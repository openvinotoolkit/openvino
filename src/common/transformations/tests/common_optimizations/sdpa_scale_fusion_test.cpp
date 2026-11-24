// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/sdpa_scale_fusion.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "ov_ops/type_relaxed.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v13 = ov::op::v13;
TEST_F(TransformationTestsF, SDPAScaleFusionTest1) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<v0::Parameter>(element::f32, value_shape);
    const auto scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{8.0f});
    const auto v_scaled = std::make_shared<v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<v1::Multiply>(query, scale_const);
        const auto k_scaled = std::make_shared<v1::Multiply>(key, scale_const);
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q_scaled, k_scaled, v_scaled, casual);

        model = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto new_mask_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
        const auto new_scale_const =
            v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{64.0f / std::sqrt(32.0f)});
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query,
                                                                           key,
                                                                           v_scaled,
                                                                           new_mask_const,
                                                                           new_scale_const,
                                                                           casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest2) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<v0::Parameter>(element::f32, value_shape);
    const auto sdpa_mask_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{2.0f});
    const auto scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{8.0f});
    const auto v_scaled = std::make_shared<v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<v1::Multiply>(query, scale_const);
        const auto k_scaled = std::make_shared<v1::Multiply>(key, scale_const);
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q_scaled,
                                                                           k_scaled,
                                                                           v_scaled,
                                                                           sdpa_mask_const,
                                                                           sdpa_scale_const,
                                                                           casual);

        model = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto new_scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{128.0f});
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query,
                                                                           key,
                                                                           v_scaled,
                                                                           sdpa_mask_const,
                                                                           new_scale_const,
                                                                           casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest3) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<v0::Parameter>(element::f32, value_shape);
    const auto sdpa_mask_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{2.0f});
    const auto scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{8.0f});
    const auto v_scaled = std::make_shared<v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<v1::Multiply>(query, scale_const);
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q_scaled,
                                                                           key,
                                                                           v_scaled,
                                                                           sdpa_mask_const,
                                                                           sdpa_scale_const,
                                                                           casual);

        model = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto new_scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{16.0f});
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query,
                                                                           key,
                                                                           v_scaled,
                                                                           sdpa_mask_const,
                                                                           new_scale_const,
                                                                           casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest4) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<v0::Parameter>(element::f32, value_shape);
    const auto sdpa_mask_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{2.0f});
    const auto scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{8.0f});
    const auto scale_dyn = std::make_shared<v0::Parameter>(element::f32, ov::Shape{});
    const auto v_scaled = std::make_shared<v1::Multiply>(value, scale_const);
    const auto casual = false;
    const auto q_scaled = std::make_shared<v1::Multiply>(query, scale_dyn);
    {
        const auto k_scaled = std::make_shared<v1::Multiply>(key, scale_const);
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q_scaled,
                                                                           k_scaled,
                                                                           v_scaled,
                                                                           sdpa_mask_const,
                                                                           sdpa_scale_const,
                                                                           casual);

        model = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, scale_dyn});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto new_scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{16.0f});
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q_scaled,
                                                                           key,
                                                                           v_scaled,
                                                                           sdpa_mask_const,
                                                                           new_scale_const,
                                                                           casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, scale_dyn});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest5) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<v0::Parameter>(element::f32, query_shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, key_shape);
    const auto value = std::make_shared<v0::Parameter>(element::f32, value_shape);
    const auto sdpa_mask_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{1.0f});
    const auto scale_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{1.0f});
    const auto scale_dyn = std::make_shared<v0::Parameter>(element::f32, ov::Shape{});
    const auto v_scaled = std::make_shared<v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<v1::Multiply>(query, scale_dyn);
        const auto k_scaled = std::make_shared<v1::Multiply>(key, scale_const);
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q_scaled,
                                                                           k_scaled,
                                                                           v_scaled,
                                                                           sdpa_mask_const,
                                                                           sdpa_scale_const,
                                                                           casual);

        model = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, scale_dyn});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto sdpa =
            std::make_shared<v13::ScaledDotProductAttention>(query, key, v_scaled, sdpa_mask_const, scale_dyn, casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, scale_dyn});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAScaleFusionTest6) {
    const PartialShape query_shape{1, 32, -1, 32};
    const PartialShape key_shape{1, 32, -1, 32};
    const PartialShape value_shape{1, 32, -1, 32};

    const auto query = std::make_shared<v0::Parameter>(element::f16, query_shape);
    const auto key = std::make_shared<v0::Parameter>(element::i8, key_shape);
    const auto value = std::make_shared<v0::Parameter>(element::f16, value_shape);
    const auto scale_const = v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{8.0f});
    const auto v_scaled = std::make_shared<v1::Multiply>(value, scale_const);
    const auto casual = false;
    {
        const auto q_scaled = std::make_shared<v1::Multiply>(query, scale_const);
        const auto k_scaled = std::make_shared<ov::op::TypeRelaxed<v1::Multiply>>(
            std::vector<element::Type>{element::f16, element::f16},
            std::vector<element::Type>{element::f16},
            ov::op::TemporaryReplaceOutputType(key, element::f16).get(),
            ov::op::TemporaryReplaceOutputType(scale_const, element::f16).get());
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q_scaled, k_scaled, v_scaled, casual);

        model = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }

    {
        const auto k_scaled_ref = std::make_shared<ov::op::TypeRelaxed<v1::Multiply>>(
            std::vector<element::Type>{element::f16, element::f16},
            std::vector<element::Type>{element::f16},
            ov::op::TemporaryReplaceOutputType(key, element::f16).get(),
            ov::op::TemporaryReplaceOutputType(scale_const, element::f16).get());
        const auto new_mask_const = v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{0.0f});
        const auto new_scale_const =
            v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{8.0f / std::sqrt(32.0f)});
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query,
                                                                           k_scaled_ref,
                                                                           v_scaled,
                                                                           new_mask_const,
                                                                           new_scale_const,
                                                                           casual);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

namespace {
// [B, S, 1, H, D] -> Squeeze(2) -> Transpose(0, 2, 1, 3) -> [B, H, S, D]
std::shared_ptr<ov::Node> to_bhsd(const Output<Node>& x) {
    const auto squeeze =
        std::make_shared<v0::Squeeze>(x, v0::Constant::create(element::i64, ov::Shape{}, std::vector<int64_t>{2}));
    return std::make_shared<v1::Transpose>(
        squeeze,
        v0::Constant::create(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3}));
}
}  // namespace

// Scalar Multiply on Q/K separated from SDPA by data-movement ops is folded into the SDPA scale;
// a single-element constant of higher rank (as produced by merging scalar multiplies) is accepted.
TEST_F(TransformationTestsF, SDPAScaleFusionThroughDataMovement) {
    const PartialShape qkv_shape{2, 16, 1, 4, 32};
    const auto query = std::make_shared<v0::Parameter>(element::f32, qkv_shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, qkv_shape);
    const auto value = std::make_shared<v0::Parameter>(element::f32, qkv_shape);
    {
        const auto q_scale = v0::Constant::create(element::f32, ov::Shape{1, 1, 1, 1, 1}, std::vector<float>{0.5f});
        const auto k_scale = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{4.0f});
        const auto q = to_bhsd(std::make_shared<v1::Multiply>(query, q_scale));
        const auto k = to_bhsd(std::make_shared<v1::Multiply>(key, k_scale));
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, k, to_bhsd(value), false);

        model = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }
    {
        const auto new_mask_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
        const auto new_scale_const =
            v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{2.0f / std::sqrt(32.0f)});
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(to_bhsd(query),
                                                                           to_bhsd(key),
                                                                           to_bhsd(value),
                                                                           new_mask_const,
                                                                           new_scale_const,
                                                                           false);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// A data-movement op shared with another consumer blocks folding for that input.
TEST_F(TransformationTestsF, SDPAScaleFusionThroughDataMovementSharedChain) {
    const PartialShape qkv_shape{2, 16, 1, 4, 32};
    const auto query = std::make_shared<v0::Parameter>(element::f32, qkv_shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, qkv_shape);
    const auto value = std::make_shared<v0::Parameter>(element::f32, qkv_shape);
    const auto scale = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.5f});
    const auto q = to_bhsd(std::make_shared<v1::Multiply>(query, scale));
    const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, to_bhsd(key), to_bhsd(value), false);
    const auto other = std::make_shared<v0::Relu>(q);

    model = std::make_shared<ov::Model>(OutputVector{sdpa, other}, ParameterVector{query, key, value});
    manager.register_pass<ov::pass::SDPAScaleFusion>();
}

// Scalar Multiply on V is merged with the scalar Multiply consuming the SDPA output; both
// disappear when the product is 1 (activations scaling: V * s and SDPA_out * 1/s).
TEST_F(TransformationTestsF, SDPAScaleFusionVScaleCancelsOutputScale) {
    const PartialShape shape{1, 8, -1, 64};
    const auto query = std::make_shared<v0::Parameter>(element::f16, shape);
    const auto key = std::make_shared<v0::Parameter>(element::f16, shape);
    const auto value = std::make_shared<v0::Parameter>(element::f16, PartialShape{1, -1, 1, 8, 64});
    const auto mask = std::make_shared<v0::Parameter>(element::f16, PartialShape{1, 8, -1, -1});
    const auto sdpa_scale = v0::Constant::create(element::f16, ov::Shape{}, std::vector<float>{0.125f});
    {
        const auto v_scaled =
            std::make_shared<v1::Multiply>(value, v0::Constant::create(element::f16, ov::Shape{}, {128.0f}));
        const auto sdpa =
            std::make_shared<v13::ScaledDotProductAttention>(query, key, to_bhsd(v_scaled), mask, sdpa_scale, false);
        const auto out =
            std::make_shared<v1::Multiply>(sdpa,
                                           v0::Constant::create(element::f16, ov::Shape{1, 1, 1, 1}, {1.0f / 128.0f}));

        model = std::make_shared<ov::Model>(OutputVector{out}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }
    {
        const auto sdpa =
            std::make_shared<v13::ScaledDotProductAttention>(query, key, to_bhsd(value), mask, sdpa_scale, false);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value, mask});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// When the product is not 1, the merged factor stays on the output Multiply.
TEST_F(TransformationTestsF, SDPAScaleFusionVScaleMergedIntoOutputScale) {
    const PartialShape shape{1, 8, -1, 64};
    const auto query = std::make_shared<v0::Parameter>(element::f32, shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, shape);
    const auto value = std::make_shared<v0::Parameter>(element::f32, shape);
    const auto mask = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, 8, -1, -1});
    {
        const auto v_scaled =
            std::make_shared<v1::Multiply>(value, v0::Constant::create(element::f32, ov::Shape{}, {4.0f}));
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query, key, v_scaled, mask, false);
        const auto out = std::make_shared<v1::Multiply>(sdpa, v0::Constant::create(element::f32, ov::Shape{}, {3.0f}));

        model = std::make_shared<ov::Model>(OutputVector{out}, ParameterVector{query, key, value, mask});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }
    {
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query, key, value, mask, false);
        const auto out = std::make_shared<v1::Multiply>(sdpa, v0::Constant::create(element::f32, ov::Shape{}, {12.0f}));
        model_ref = std::make_shared<ov::Model>(OutputVector{out}, ParameterVector{query, key, value, mask});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// The scale may be either operand of the Multiply, also behind data-movement ops.
TEST_F(TransformationTestsF, SDPAScaleFusionSwappedOperands) {
    const PartialShape shape{2, 4, 16, 32};
    const auto query = std::make_shared<v0::Parameter>(element::f32, shape);
    const auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape{2, 16, 1, 4, 32});
    const auto value = std::make_shared<v0::Parameter>(element::f32, shape);
    {
        const auto q = std::make_shared<v1::Multiply>(v0::Constant::create(element::f32, ov::Shape{}, {0.5f}), query);
        const auto k =
            to_bhsd(std::make_shared<v1::Multiply>(v0::Constant::create(element::f32, ov::Shape{1}, {4.0f}), key));
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, k, value, false);

        model = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
        manager.register_pass<ov::pass::SDPAScaleFusion>();
    }
    {
        const auto new_mask_const = v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{0.0f});
        const auto new_scale_const =
            v0::Constant::create(element::f32, ov::Shape{}, std::vector<float>{2.0f / std::sqrt(32.0f)});
        const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query,
                                                                           to_bhsd(key),
                                                                           value,
                                                                           new_mask_const,
                                                                           new_scale_const,
                                                                           false);
        model_ref = std::make_shared<ov::Model>(OutputVector{sdpa}, ParameterVector{query, key, value});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// With dynamic ranks a [1,1,1,1,1] output scale may raise the output rank, so the output Multiply
// must be kept even though the V and output scales cancel.
TEST_F(TransformationTestsF, SDPAScaleFusionVScaleDynamicRankKept) {
    const auto query = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto value = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto v_scaled =
        std::make_shared<v1::Multiply>(value, v0::Constant::create(element::f32, ov::Shape{}, {2.0f}));
    const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query, key, v_scaled, false);
    const auto out =
        std::make_shared<v1::Multiply>(sdpa, v0::Constant::create(element::f32, ov::Shape{1, 1, 1, 1, 1}, {0.5f}));

    model = std::make_shared<ov::Model>(OutputVector{out}, ParameterVector{query, key, value});
    manager.register_pass<ov::pass::SDPAScaleFusion>();
}

namespace {
// SDPA(Q, K, V * v_scale) * out_scale in f16 with one key, so that the output equals the scaled V.
std::shared_ptr<ov::Model> make_f16_v_and_output_scaled(float v_scale, float out_scale) {
    const auto query = std::make_shared<v0::Parameter>(element::f16, PartialShape{1, 2, 4, 16});
    const auto key = std::make_shared<v0::Parameter>(element::f16, PartialShape{1, 2, 1, 16});
    const auto value = std::make_shared<v0::Parameter>(element::f16, PartialShape{1, 2, 1, 16});
    const auto v_scaled =
        std::make_shared<v1::Multiply>(value, v0::Constant::create(element::f16, ov::Shape{}, {v_scale}));
    const auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(query, key, v_scaled, false);
    const auto out = std::make_shared<v1::Multiply>(sdpa, v0::Constant::create(element::f16, ov::Shape{}, {out_scale}));
    return std::make_shared<ov::Model>(OutputVector{out}, ParameterVector{query, key, value});
}
}  // namespace

// 256 * 256 is not representable in f16 (max 65504): merging would produce an infinite scale although
// e.g. V = 2^-13 gives the finite result 8 in the original graph. The graph must stay unchanged.
TEST_F(TransformationTestsF, SDPAScaleFusionVScaleF16OverflowKept) {
    model = make_f16_v_and_output_scaled(256.0f, 256.0f);
    manager.register_pass<ov::pass::SDPAScaleFusion>();
}

// 2^-13 * 2^-13 flushes to zero in f16 although e.g. V = 4096 gives 2^-14 in the original graph.
TEST_F(TransformationTestsF, SDPAScaleFusionVScaleF16UnderflowKept) {
    model = make_f16_v_and_output_scaled(std::ldexp(1.0f, -13), std::ldexp(1.0f, -13));
    manager.register_pass<ov::pass::SDPAScaleFusion>();
}
