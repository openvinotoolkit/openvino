// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/rewrite_gqa_attention_mask.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/ops.hpp"

namespace {

using namespace ov;

constexpr int64_t kKvLen = 8;
constexpr int64_t kQLen = 4;
constexpr int64_t kPast = kKvLen - kQLen;

struct MaskModelParams {
    bool rank4 = false;        // [1, 1, Q, KV] mask instead of [Q, KV]
    bool query_first = false;  // Less(query, key) instead of Less(key, query)
    bool with_attention_mask = true;
    std::string select_name = "GroupQueryAttention_0/Select";
};

// Builds:
//   CumSum(K) ------------------------------------------+
//   CumSum(Q) -> [Add] -> Broadcast -> Transpose --> Less|LessEqual -> [LogicalAnd(attention_mask)] -> Select
//   -> Add(logits) -> Softmax
std::shared_ptr<Model> build_model(const MaskModelParams& p, bool rewritten) {
    ParameterVector params;

    auto attention_mask = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, kKvLen});
    attention_mask->output(0).set_names({p.with_attention_mask ? "attention_mask" : "other_input"});
    params.push_back(attention_mask);

    const Shape k_shape = p.rank4 ? Shape{1, 1, 1, static_cast<size_t>(kKvLen)} : Shape{1, static_cast<size_t>(kKvLen)};
    auto k_src = std::make_shared<op::v0::Parameter>(element::i64, k_shape);
    auto q_src = std::make_shared<op::v0::Parameter>(element::i64, Shape{static_cast<size_t>(kQLen)});
    params.push_back(k_src);
    params.push_back(q_src);

    auto k_axis = op::v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(k_shape.size()) - 1});
    auto k_cumsum = std::make_shared<op::v0::CumSum>(k_src, k_axis);

    auto q_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    std::shared_ptr<Node> q_value = std::make_shared<op::v0::CumSum>(q_src, q_axis);
    if (!rewritten) {
        q_value = std::make_shared<op::v1::Add>(q_value, op::v0::Constant::create(element::i64, Shape{}, {kPast}));
    }

    const std::vector<int64_t> bcast_shape =
        p.rank4 ? std::vector<int64_t>{1, 1, 1, kQLen} : std::vector<int64_t>{1, kQLen};
    auto bcast = std::make_shared<op::v3::Broadcast>(
        q_value,
        op::v0::Constant::create(element::i64, Shape{bcast_shape.size()}, bcast_shape));
    const std::vector<int64_t> order = p.rank4 ? std::vector<int64_t>{0, 1, 3, 2} : std::vector<int64_t>{1, 0};
    auto transpose =
        std::make_shared<op::v1::Transpose>(bcast, op::v0::Constant::create(element::i64, Shape{order.size()}, order));

    Output<Node> lhs = k_cumsum, rhs = transpose;
    if (p.query_first) {
        std::swap(lhs, rhs);
    }
    std::shared_ptr<Node> cond;
    if (rewritten) {
        cond = std::make_shared<op::v1::LessEqual>(lhs, rhs);
        std::shared_ptr<Node> mask = std::make_shared<op::v0::Convert>(attention_mask, element::boolean);
        if (p.rank4) {
            mask = std::make_shared<op::v0::Unsqueeze>(mask, op::v0::Constant::create(element::i64, Shape{2}, {1, 2}));
        }
        cond = std::make_shared<op::v1::LogicalAnd>(cond, mask);
    } else {
        cond = std::make_shared<op::v1::Less>(lhs, rhs);
    }

    auto zero = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto lowest = op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()});
    auto select = std::make_shared<op::v1::Select>(cond, zero, lowest);
    select->set_friendly_name(p.select_name);

    auto logits = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, kQLen, kKvLen});
    params.push_back(logits);
    auto add = std::make_shared<op::v1::Add>(logits, select);
    auto softmax = std::make_shared<op::v8::Softmax>(add, -1);

    return std::make_shared<Model>(OutputVector{softmax}, params);
}

class RewriteGQAAttentionMaskTest : public TransformationTestsF,
                                    public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(RewriteGQAAttentionMaskTest, RewritesMask) {
    MaskModelParams p;
    std::tie(p.rank4, p.query_first) = GetParam();
    model = build_model(p, false);
    manager.register_pass<npuw::RewriteGQAAttentionMask>();
    model_ref = build_model(p, true);
}

INSTANTIATE_TEST_SUITE_P(NPUW,
                         RewriteGQAAttentionMaskTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()),
                         [](const ::testing::TestParamInfo<std::tuple<bool, bool>>& info) {
                             return std::string(std::get<0>(info.param) ? "Rank4" : "Rank2") +
                                    (std::get<1>(info.param) ? "_QueryFirst" : "_KeyFirst");
                         });

TEST_F(TransformationTestsF, RewriteGQAAttentionMask_NotAppliedWithoutGQAName) {
    MaskModelParams p;
    p.select_name = "SomeOtherAttention/Select";
    model = build_model(p, false);
    manager.register_pass<npuw::RewriteGQAAttentionMask>();
    model_ref = build_model(p, false);
}

TEST_F(TransformationTestsF, RewriteGQAAttentionMask_NotAppliedWithoutAttentionMask) {
    MaskModelParams p;
    p.with_attention_mask = false;
    model = build_model(p, false);
    manager.register_pass<npuw::RewriteGQAAttentionMask>();
    model_ref = build_model(p, false);
}

}  // namespace
