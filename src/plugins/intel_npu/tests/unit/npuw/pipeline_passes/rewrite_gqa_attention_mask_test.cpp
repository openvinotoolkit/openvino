// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/rewrite_gqa_attention_mask.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/constant_folding.hpp"

namespace {

using namespace ov;

constexpr int64_t kKvLen = 8;
constexpr int64_t kQLen = 4;

struct MaskModelParams {
    int64_t kv_len = kKvLen;
    int64_t q_len = kQLen;
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

    auto attention_mask = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, p.kv_len});
    attention_mask->output(0).set_names({p.with_attention_mask ? "attention_mask" : "other_input"});
    params.push_back(attention_mask);

    const Shape k_shape =
        p.rank4 ? Shape{1, 1, 1, static_cast<size_t>(p.kv_len)} : Shape{1, static_cast<size_t>(p.kv_len)};
    auto k_src = std::make_shared<op::v0::Parameter>(element::i64, k_shape);
    auto q_src = std::make_shared<op::v0::Parameter>(element::i64, Shape{static_cast<size_t>(p.q_len)});
    params.push_back(k_src);
    params.push_back(q_src);

    auto k_axis = op::v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(k_shape.size()) - 1});
    auto k_cumsum = std::make_shared<op::v0::CumSum>(k_src, k_axis);

    auto q_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    std::shared_ptr<Node> q_value = std::make_shared<op::v0::CumSum>(q_src, q_axis);
    if (!rewritten) {
        q_value = std::make_shared<op::v1::Add>(q_value, op::v0::Constant::create(element::i64, Shape{}, {123}));
    }

    const std::vector<int64_t> bcast_shape =
        p.rank4 ? std::vector<int64_t>{1, 1, 1, p.q_len} : std::vector<int64_t>{1, p.q_len};
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
        cond = p.q_len == 1 ? mask : std::make_shared<op::v1::LogicalAnd>(cond, mask);
    } else {
        cond = std::make_shared<op::v1::Less>(lhs, rhs);
    }

    auto zero = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto lowest = op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()});
    auto select = std::make_shared<op::v1::Select>(cond, zero, lowest);
    select->set_friendly_name(p.select_name);

    auto logits =
        std::make_shared<op::v0::Parameter>(element::f32,
                                            Shape{1, 2, static_cast<size_t>(p.q_len), static_cast<size_t>(p.kv_len)});
    params.push_back(logits);
    auto add = std::make_shared<op::v1::Add>(logits, select);
    auto softmax = std::make_shared<op::v8::Softmax>(add, -1);

    return std::make_shared<Model>(OutputVector{softmax}, params);
}

class RewriteGQAAttentionMaskTest : public TransformationTestsF,
                                    public ::testing::WithParamInterface<std::tuple<bool, bool, int64_t>> {};

TEST_P(RewriteGQAAttentionMaskTest, RewritesMask) {
    MaskModelParams p;
    std::tie(p.rank4, p.query_first, p.q_len) = GetParam();
    model = build_model(p, false);
    manager.register_pass<npuw::RewriteGQAAttentionMask>();
    manager.register_pass<pass::ConstantFolding>();
    comparator.enable(FunctionsComparator::CONST_VALUES);
    model_ref = build_model(p, true);
}

INSTANTIATE_TEST_SUITE_P(NPUW,
                         RewriteGQAAttentionMaskTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Values(1, 4)),
                         [](const ::testing::TestParamInfo<std::tuple<bool, bool, int64_t>>& info) {
                             return std::string(std::get<0>(info.param) ? "Rank4" : "Rank2") +
                                    (std::get<1>(info.param) ? "_QueryFirst" : "_KeyFirst") + "_Q" +
                                    std::to_string(std::get<2>(info.param));
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

class RewriteGQAAttentionMaskValuesTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t, bool, int64_t>> {
};

TEST_P(RewriteGQAAttentionMaskValuesTest, StageMaskPreservesValidKeysAndCausality) {
    MaskModelParams p;
    int64_t reshape_delta = 0;
    std::tie(p.kv_len, p.q_len, p.rank4, reshape_delta) = GetParam();
    auto model = build_model(p, false);
    npuw::RewriteGQAAttentionMask().run_on_model(model);

    std::shared_ptr<Node> condition;
    for (const auto& node : model->get_ordered_ops()) {
        if (is_type<op::v1::Select>(node)) {
            condition = node->input_value(0).get_node_shared_ptr();
        }
    }
    ASSERT_NE(condition, nullptr);
    ASSERT_EQ(is_type<op::v1::LogicalAnd>(condition), p.q_len != 1);

    // Changing key capacity must preserve the selected mask behavior.
    const int64_t key_length = p.kv_len + reshape_delta;
    auto key_shape = model->get_parameters()[1]->get_shape();
    key_shape.back() = key_length;
    model->get_parameters()[0]->set_partial_shape(PartialShape{1, key_length});
    model->get_parameters()[1]->set_partial_shape(key_shape);
    auto logits_shape = model->get_parameters()[3]->get_shape();
    logits_shape.back() = key_length;
    model->get_parameters()[3]->set_partial_shape(logits_shape);
    model->validate_nodes_and_infer_types();
    auto mask_model = std::make_shared<Model>(OutputVector{condition}, model->get_parameters());

    TensorVector inputs;
    for (const auto& parameter : model->get_parameters()) {
        inputs.emplace_back(parameter->get_element_type(), parameter->get_shape());
        auto& tensor = inputs.back();
        if (tensor.get_element_type() == element::i64) {
            std::fill_n(tensor.data<int64_t>(), tensor.get_size(), 1);
        } else {
            std::fill_n(tensor.data<float>(), tensor.get_size(), 0.f);
        }
    }
    for (bool padded : {false, true}) {
        // Keep past tokens on the left and current tokens on the right, with an invalid KV gap.
        inputs[0].data<int64_t>()[2] = padded ? 0 : 1;
        inputs[0].data<int64_t>()[3] = padded ? 0 : 1;
        const auto outputs = ov::test::utils::infer_on_template(mask_model, inputs);
        ASSERT_EQ(outputs[0].get_size(), static_cast<size_t>(p.q_len * key_length));
        for (int64_t q = 0; q < p.q_len; ++q) {
            for (int64_t k = 0; k < key_length; ++k) {
                const bool causal = p.q_len == 1 || k <= q;
                const bool expected = causal && inputs[0].data<int64_t>()[k] != 0;
                EXPECT_EQ(outputs[0].data<char>()[q * key_length + k] != 0, expected)
                    << "q=" << q << ", k=" << k << ", padded=" << padded;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(NPUW,
                         RewriteGQAAttentionMaskValuesTest,
                         ::testing::Combine(::testing::Values(8, 1152),
                                            ::testing::Values(1, 4, 8),
                                            ::testing::Bool(),
                                            ::testing::Values(0, 2)));

}  // namespace
