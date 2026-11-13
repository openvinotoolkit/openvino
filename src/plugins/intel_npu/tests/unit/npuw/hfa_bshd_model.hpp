// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/ops.hpp"

namespace ov {
namespace test {
namespace npuw {

// Residual chain of decomposed-attention layers in the form the llama.cpp OpenVINO backend emits
// for HFA prefill: past K/V are f16 Parameters, optionally in [B, S, H, D] layout followed by a
// {0,2,1,3} Transpose right before the KV Concat, and the attention history length is a model input.
struct HfaBshdModelConfig {
    std::size_t num_layers = 2;
    std::size_t num_heads = 4;
    std::size_t num_kv_heads = 2;
    std::size_t head_dim = 16;
    std::size_t query_len = 16;
    std::size_t past_len = 48;
    bool bshd_key = true;
    bool bshd_value = true;
};

inline const std::string kHfaHistoryInputName = "history_size";
inline const std::string kHfaMaskInputName = "attention_mask";
inline const std::string kHfaHiddenOutputName = "hidden_out";

inline std::shared_ptr<ov::Model> build_hfa_bshd_model(const HfaBshdModelConfig& cfg = {}) {
    using namespace ov;

    const auto H = cfg.num_heads;
    const auto Hkv = cfg.num_kv_heads;
    const auto D = cfg.head_dim;
    const auto Q = cfg.query_len;
    const auto P = cfg.past_len;
    const auto context = P + Q;
    auto i64_const = [](const std::vector<int64_t>& values) {
        return op::v0::Constant::create(element::i64, Shape{values.size()}, values);
    };
    auto to_i64 = [](std::size_t v) {
        return static_cast<int64_t>(v);
    };

    auto hidden_in = ov::test::utils::make_param(element::f32, Shape{1, Q, H * D}, "hidden");
    auto mask = ov::test::utils::make_param(element::f32, Shape{1, 1, Q, context}, kHfaMaskInputName);
    auto history = ov::test::utils::make_param(element::i64, Shape{1}, kHfaHistoryInputName);
    ParameterVector params{hidden_in, mask, history};

    auto make_past = [&](const std::string& name, bool bshd) -> Output<Node> {
        const Shape shape = bshd ? Shape{1, P, Hkv, D} : Shape{1, Hkv, P, D};
        auto param = ov::test::utils::make_param(element::f16, shape, name);
        params.push_back(param);
        Output<Node> out = std::make_shared<op::v0::Convert>(param, element::f32);
        if (bshd) {
            out = std::make_shared<op::v1::Transpose>(out, i64_const({0, 2, 1, 3}));
        }
        return out;
    };

    auto expand_gqa = [&](const Output<Node>& kv) -> Output<Node> {
        if (Hkv == H) {
            return kv;
        }
        auto unsq = std::make_shared<op::v0::Unsqueeze>(kv, i64_const({2}));
        auto bcast = std::make_shared<op::v3::Broadcast>(
            unsq,
            i64_const({1, to_i64(Hkv), to_i64(H / Hkv), to_i64(context), to_i64(D)}));
        return std::make_shared<op::v1::Reshape>(bcast, i64_const({1, to_i64(H), to_i64(context), to_i64(D)}), false);
    };

    Output<Node> hidden = hidden_in;
    for (std::size_t n = 0; n < cfg.num_layers; ++n) {
        const auto idx = std::to_string(n);
        auto new_key = ov::test::utils::make_param(element::f32, Shape{1, Hkv, Q, D}, "new_key." + idx);
        auto new_value = ov::test::utils::make_param(element::f32, Shape{1, Hkv, Q, D}, "new_value." + idx);
        params.insert(params.end(), {new_key, new_value});

        auto query = std::make_shared<op::v1::Transpose>(
            std::make_shared<op::v1::Reshape>(hidden, i64_const({1, to_i64(Q), to_i64(H), to_i64(D)}), false),
            i64_const({0, 2, 1, 3}));
        auto key_concat =
            std::make_shared<op::v0::Concat>(OutputVector{make_past("past_key." + idx, cfg.bshd_key), new_key}, 2);
        auto value_concat =
            std::make_shared<op::v0::Concat>(OutputVector{make_past("past_value." + idx, cfg.bshd_value), new_value},
                                             2);

        auto qk = std::make_shared<op::v0::MatMul>(query, expand_gqa(key_concat), false, true);
        auto add = std::make_shared<op::v1::Add>(qk, mask);
        auto softmax = std::make_shared<op::v8::Softmax>(add, -1);
        auto sv = std::make_shared<op::v0::MatMul>(softmax, expand_gqa(value_concat));
        auto transpose = std::make_shared<op::v1::Transpose>(sv, i64_const({0, 2, 1, 3}));
        auto attn = std::make_shared<op::v1::Reshape>(transpose, i64_const({1, to_i64(Q), to_i64(H * D)}), false);
        hidden = std::make_shared<op::v1::Add>(hidden, attn);
    }

    auto hidden_result = std::make_shared<op::v0::Result>(hidden);
    hidden_result->output(0).get_tensor().set_names({kHfaHiddenOutputName});

    // Keep the history input alive the way a real consumer (e.g. RoPE positions) would.
    auto history_use = std::make_shared<op::v1::Multiply>(std::make_shared<op::v0::Convert>(history, element::f32),
                                                          op::v0::Constant::create(element::f32, Shape{1}, {2.0f}));
    auto history_result = std::make_shared<op::v0::Result>(history_use);
    history_result->output(0).get_tensor().set_names({"history_echo"});

    auto model = std::make_shared<Model>(ResultVector{hidden_result, history_result}, params, "hfa_bshd_model");
    model->validate_nodes_and_infer_types();
    return model;
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
