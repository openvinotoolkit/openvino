// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/sdpa_select_mask_fusion.hpp"

#include <limits>
#include <memory>
#include <vector>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {
// Sentinel a framework produces for masked_fill(-inf) once the mask runs in half precision, and
// the additive mask values the pass emits: half of the lowest value of the score precision.
const float NEG_INF = static_cast<float>(std::numeric_limits<ov::float16>::lowest());
const float MASK_F16 = NEG_INF / 2.0f;
const float MASK_BF16 = static_cast<float>(std::numeric_limits<ov::bfloat16>::lowest()) / 2.0f;

constexpr int64_t H = 4;
constexpr int64_t S = 8;
constexpr int64_t D = 16;
const ov::Shape q_shape{1, H, S, D};
const ov::Shape k_shape{1, H, D, S};
const ov::Shape v_shape{1, H, S, D};
const ov::Shape scores_shape{1, H, S, S};
const ov::Shape cond_shape{1, 1, S, S};

// Decomposed attention front half: q, k, v, cond parameters and scores = MatMul(Q, K).
struct Attention {
    ov::ParameterVector params;
    std::shared_ptr<ov::Node> scores;
    std::shared_ptr<ov::op::v0::Parameter> cond;
    std::shared_ptr<ov::op::v0::Parameter> value;
};

Attention make_attention(const ov::element::Type& et, const ov::Shape& c_shape = cond_shape) {
    auto q = std::make_shared<ov::op::v0::Parameter>(et, q_shape);
    auto k = std::make_shared<ov::op::v0::Parameter>(et, k_shape);
    auto v = std::make_shared<ov::op::v0::Parameter>(et, v_shape);
    auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, c_shape);
    auto scores = std::make_shared<ov::op::v0::MatMul>(q, k);
    return Attention{ov::ParameterVector{q, k, v, cond}, scores, cond, v};
}

std::shared_ptr<ov::Model> finish_attention(const Attention& a, const ov::Output<ov::Node>& masked) {
    auto softmax = std::make_shared<ov::op::v8::Softmax>(masked, -1);
    auto out = std::make_shared<ov::op::v0::MatMul>(softmax, a.value);
    return std::make_shared<ov::Model>(ov::OutputVector{out}, a.params);
}

void enable_full_compare(FunctionsComparator& comparator) {
    comparator.enable(FunctionsComparator::CONST_VALUES);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}
}  // namespace

TEST_F(TransformationTestsF, SDPASelectMaskFusion_Fp16SelectToAdd) {
    {
        auto a = make_attention(ov::element::f16);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, a.scores, neg_inf);
        model = finish_attention(a, select);
        manager.register_pass<SDPASelectMaskFusion>();
    }
    {
        auto a = make_attention(ov::element::f16);
        auto zero = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.0f});
        auto mask_value = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {MASK_F16});
        auto add_mask = std::make_shared<ov::op::v1::Select>(a.cond, zero, mask_value);
        auto add = std::make_shared<ov::op::v1::Add>(a.scores, add_mask);
        model_ref = finish_attention(a, add);
        enable_full_compare(comparator);
    }
}

TEST_F(TransformationTestsF, SDPASelectMaskFusion_Bf16SelectToAdd) {
    {
        auto a = make_attention(ov::element::bf16);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::bf16, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, a.scores, neg_inf);
        model = finish_attention(a, select);
        manager.register_pass<SDPASelectMaskFusion>();
    }
    {
        auto a = make_attention(ov::element::bf16);
        auto zero = ov::op::v0::Constant::create(ov::element::bf16, ov::Shape{}, {0.0f});
        auto mask_value = ov::op::v0::Constant::create(ov::element::bf16, ov::Shape{}, {MASK_BF16});
        auto add_mask = std::make_shared<ov::op::v1::Select>(a.cond, zero, mask_value);
        auto add = std::make_shared<ov::op::v1::Add>(a.scores, add_mask);
        model_ref = finish_attention(a, add);
        enable_full_compare(comparator);
    }
}

TEST_F(TransformationTestsF, SDPASelectMaskFusion_SelectThroughReshapeToSoftmax) {
    const std::vector<int64_t> new_shape{1, H, S, S};
    {
        auto a = make_attention(ov::element::f16);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, a.scores, neg_inf);
        auto rs_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, new_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(select, rs_c, false);
        model = finish_attention(a, reshape);
        manager.register_pass<SDPASelectMaskFusion>();
    }
    {
        auto a = make_attention(ov::element::f16);
        auto zero = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.0f});
        auto mask_value = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {MASK_F16});
        auto add_mask = std::make_shared<ov::op::v1::Select>(a.cond, zero, mask_value);
        auto add = std::make_shared<ov::op::v1::Add>(a.scores, add_mask);
        auto rs_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, new_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(add, rs_c, false);
        model_ref = finish_attention(a, reshape);
        enable_full_compare(comparator);
    }
}

// Production shape: the mask is applied after the Q*K scaling, so Q*K sits behind a Multiply.
TEST_F(TransformationTestsF, SDPASelectMaskFusion_ScaledScoresSelectToAdd) {
    auto scale_const = [] {
        return ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.125f});
    };
    {
        auto a = make_attention(ov::element::f16);
        auto scaled = std::make_shared<ov::op::v1::Multiply>(a.scores, scale_const());
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, scaled, neg_inf);
        model = finish_attention(a, select);
        manager.register_pass<SDPASelectMaskFusion>();
    }
    {
        auto a = make_attention(ov::element::f16);
        auto scaled = std::make_shared<ov::op::v1::Multiply>(a.scores, scale_const());
        auto zero = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.0f});
        auto mask_value = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {MASK_F16});
        auto add_mask = std::make_shared<ov::op::v1::Select>(a.cond, zero, mask_value);
        auto add = std::make_shared<ov::op::v1::Add>(scaled, add_mask);
        model_ref = finish_attention(a, add);
        enable_full_compare(comparator);
    }
}

// f32 keeps its Select: the fused SDPA primitive only supports f16/bf16, so rewriting it would only
// add an eltwise that cannot fuse.
TEST_F(TransformationTestsF, SDPASelectMaskFusion_F32NotRewritten) {
    auto build = [] {
        auto a = make_attention(ov::element::f32);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, a.scores, neg_inf);
        return finish_attention(a, select);
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

// A finite sentinel that does not saturate -inf keeps its Select: with scores below the sentinel the
// additive form would change the result, e.g. Softmax([-20000, -10000]) = [0, 1] but
// Softmax([-20000, masked]) = [1, 0].
TEST_F(TransformationTestsF, SDPASelectMaskFusion_FiniteSentinel_NoChange) {
    auto build = [] {
        auto a = make_attention(ov::element::f16);
        auto small = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {-10000.0f});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, a.scores, small);
        return finish_attention(a, select);
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

TEST_F(TransformationTestsF, SDPASelectMaskFusion_NoSoftmaxConsumer_NoChange) {
    auto build = [] {
        auto a = make_attention(ov::element::f16);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, a.scores, neg_inf);
        auto relu = std::make_shared<ov::op::v0::Relu>(select);
        return std::make_shared<ov::Model>(ov::OutputVector{relu}, a.params);
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

// A plain Select -> Softmax without the attention MatMuls keeps its Select, so unrelated graphs do
// not pay for an extra eltwise.
TEST_F(TransformationTestsF, SDPASelectMaskFusion_NonAttentionSelectSoftmax_NoChange) {
    auto build = [] {
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, scores_shape);
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(cond, scores, neg_inf);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(select, -1);
        return std::make_shared<ov::Model>(ov::OutputVector{softmax}, ov::ParameterVector{scores, cond});
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

TEST_F(TransformationTestsF, SDPASelectMaskFusion_SoftmaxWithoutValueMatMul_NoChange) {
    auto build = [] {
        auto a = make_attention(ov::element::f16);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, a.scores, neg_inf);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(select, -1);
        return std::make_shared<ov::Model>(ov::OutputVector{softmax}, a.params);
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

TEST_F(TransformationTestsF, SDPASelectMaskFusion_NonScalarElseValue_NoChange) {
    auto build = [] {
        auto a = make_attention(ov::element::f16);
        std::vector<float> vals(S, NEG_INF);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(S)}, vals);
        auto select = std::make_shared<ov::op::v1::Select>(a.cond, a.scores, neg_inf);
        return finish_attention(a, select);
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

// PDPD broadcasting with a lower-rank condition: splitting the Select into a mask Select plus an Add
// cannot reproduce the explicit PDPD axis.
TEST_F(TransformationTestsF, SDPASelectMaskFusion_PdpdBroadcast_NoChange) {
    auto build = [] {
        auto a = make_attention(ov::element::f16, ov::Shape{S, S});
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(a.cond,
                                                           a.scores,
                                                           neg_inf,
                                                           ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 2));
        return finish_attention(a, select);
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

// NONE broadcasting requires all three inputs to share one shape, which the scalar mask constants of
// the additive form cannot satisfy.
TEST_F(TransformationTestsF, SDPASelectMaskFusion_NoneBroadcast_NoChange) {
    auto build = [] {
        auto a = make_attention(ov::element::f16, scores_shape);
        std::vector<float> vals(shape_size(scores_shape), NEG_INF);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f16, scores_shape, vals);
        auto select = std::make_shared<ov::op::v1::Select>(a.cond,
                                                           a.scores,
                                                           neg_inf,
                                                           ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NONE));
        return finish_attention(a, select);
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
