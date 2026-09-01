// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
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
constexpr float NEG_INF = -65504.0f;
const float NEG_INF_ADD = -std::numeric_limits<float>::infinity();

// scores [1, H, S, S], boolean where-mask cond broadcastable as [1, 1, S, S]
constexpr int64_t H = 4;
constexpr int64_t S = 8;
const ov::Shape scores_shape{1, H, S, S};
const ov::Shape cond_shape{1, 1, S, S};
}  // namespace

// ---------------------------------------------------------------------------
// 1. Select(cond, scores, neg_inf) -> Softmax
//    => Add(scores, Select(cond, 0, neg_inf)) -> Softmax
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPASelectMaskFusion_BasicSelectToAdd) {
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(cond, scores, neg_inf);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(select, -1);

        model = std::make_shared<ov::Model>(ov::OutputVector{softmax}, ov::ParameterVector{scores, cond});
        manager.register_pass<SDPASelectMaskFusion>();
    }
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {NEG_INF_ADD});
        auto add_mask = std::make_shared<ov::op::v1::Select>(cond, zero, neg_inf);
        auto add = std::make_shared<ov::op::v1::Add>(scores, add_mask);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(add, -1);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{softmax}, ov::ParameterVector{scores, cond});
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 2. Select feeding Softmax through a single Reshape is still converted.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPASelectMaskFusion_SelectThroughReshapeToSoftmax) {
    const std::vector<int64_t> new_shape{1, H, S, S};
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(cond, scores, neg_inf);
        auto rs_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, new_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(select, rs_c, false);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(reshape, -1);

        model = std::make_shared<ov::Model>(ov::OutputVector{softmax}, ov::ParameterVector{scores, cond});
        manager.register_pass<SDPASelectMaskFusion>();
    }
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {NEG_INF_ADD});
        auto add_mask = std::make_shared<ov::op::v1::Select>(cond, zero, neg_inf);
        auto add = std::make_shared<ov::op::v1::Add>(scores, add_mask);
        auto rs_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, new_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(add, rs_c, false);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(reshape, -1);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{softmax}, ov::ParameterVector{scores, cond});
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 3. Negative: else-value magnitude too small to act as -inf (> -1e4) -> no change.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPASelectMaskFusion_ElseValueTooSmall_NoChange) {
    auto build = [] {
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto small = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-5.0f});
        auto select = std::make_shared<ov::op::v1::Select>(cond, scores, small);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(select, -1);
        return std::make_shared<ov::Model>(ov::OutputVector{softmax}, ov::ParameterVector{scores, cond});
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

// ---------------------------------------------------------------------------
// 4. Negative: Select does not feed a Softmax -> no change (rewrite not equivalent).
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPASelectMaskFusion_NoSoftmaxConsumer_NoChange) {
    auto build = [] {
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {NEG_INF});
        auto select = std::make_shared<ov::op::v1::Select>(cond, scores, neg_inf);
        auto relu = std::make_shared<ov::op::v0::Relu>(select);
        return std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{scores, cond});
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

// ---------------------------------------------------------------------------
// 5. Negative: non-scalar else-value -> no change.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPASelectMaskFusion_NonScalarElseValue_NoChange) {
    auto build = [] {
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        std::vector<float> vals(S, NEG_INF);
        auto neg_inf = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(S)}, vals);
        auto select = std::make_shared<ov::op::v1::Select>(cond, scores, neg_inf);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(select, -1);
        return std::make_shared<ov::Model>(ov::OutputVector{softmax}, ov::ParameterVector{scores, cond});
    };
    model = build();
    manager.register_pass<SDPASelectMaskFusion>();
    model_ref = build();
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
