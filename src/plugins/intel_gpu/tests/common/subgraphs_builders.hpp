// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include "openvino/core/dimension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/make_stateful.hpp"

namespace tests {

inline std::shared_ptr<ov::Node> make_state_initializer(ov::Output<ov::Node> input,
                                                        ov::element::Type_t element_type,
                                                        ov::PartialShape kv_cache_size,
                                                        std::vector<int64_t> qkv_order) {
    auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i32);

    auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, qkv_order[0]);
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto gather = std::make_shared<ov::op::v8::Gather>(shapeof, indices, axis, 0);

    auto bcast_value = std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape{}, 0.0f);
    ov::NodeVector dims(kv_cache_size.size(), nullptr);

    for (size_t i = 0; i < kv_cache_size.size(); i++) {
        if (i == static_cast<size_t>(qkv_order[0])) {
            dims[i] = gather;
        } else {
            dims[i] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, static_cast<int32_t>(kv_cache_size[i].get_min_length()));
        }
    }

    ov::NodeVector ordered_dims = dims;
    for (size_t i = 0; i < dims.size(); i++) {
        ordered_dims[i] = dims[qkv_order[i]];
    }
    auto shape = std::make_shared<ov::op::v0::Concat>(dims, 0);
    return std::make_shared<ov::op::v3::Broadcast>(bcast_value, shape);
}

inline std::shared_ptr<ov::Node> make_attention_mask(ov::Output<ov::Node> q,
                                                     ov::Output<ov::Node> k,
                                                     ov::element::Type_t element_type,
                                                     std::vector<int64_t> qkv_order) {
    std::shared_ptr<ov::Node> minus_inf = nullptr;
    if (element_type == ov::element::f32)
        minus_inf = ov::op::v0::Constant::create(element_type, ov::Shape{}, {-std::numeric_limits<float>::infinity()});
    else if (element_type == ov::element::f16)
        minus_inf = ov::op::v0::Constant::create(element_type, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()});
    auto q_shape = std::make_shared<ov::op::v3::ShapeOf>(q, ov::element::i32);
    auto k_shape = std::make_shared<ov::op::v3::ShapeOf>(k, ov::element::i32);

    auto seq_len_idx = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {qkv_order[2]});
    auto zero_i = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto one_i = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto zero_f = std::make_shared<ov::op::v1::ConvertLike>(zero_i, q);

    auto target_s_len = std::make_shared<ov::op::v8::Gather>(q_shape, seq_len_idx, zero_i);
    auto source_s_len = std::make_shared<ov::op::v8::Gather>(k_shape, seq_len_idx, zero_i);
    auto ssl = std::make_shared<ov::op::v0::Unsqueeze>(source_s_len, zero_i);
    auto tsl = std::make_shared<ov::op::v0::Unsqueeze>(target_s_len, zero_i);
    auto mask_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{tsl, ssl}, 0);
    auto m = std::make_shared<ov::op::v1::Broadcast>(minus_inf, mask_shape);
    auto horizontal_range = std::make_shared<ov::op::v4::Range>(zero_i, source_s_len, one_i, ov::element::i32)->output(0);
    horizontal_range = std::make_shared<ov::op::v0::Unsqueeze>(horizontal_range, zero_i);
    auto stop = std::make_shared<ov::op::v1::Add>(target_s_len, one_i);
    auto vertical_range = std::make_shared<ov::op::v4::Range>(one_i, stop, one_i, ov::element::i32)->output(0);
    vertical_range = std::make_shared<ov::op::v0::Unsqueeze>(vertical_range, one_i);
    auto triu = std::make_shared<ov::op::v1::GreaterEqual>(horizontal_range, vertical_range);
    auto select = std::make_shared<ov::op::v1::Select>(triu, m, zero_f);
    auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1});
    return std::make_shared<ov::op::v0::Unsqueeze>(select, unsqueeze_axis);
}

inline std::shared_ptr<ov::Node> make_gqa(ov::Output<ov::Node> kv, size_t num_groups, std::vector<int32_t> target_shape_v, int32_t n_heads) {
    int32_t num_heads_idx = std::distance(target_shape_v.begin(), std::find(target_shape_v.begin(), target_shape_v.end(), n_heads)) + 1;
    auto unsqueeze_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, num_heads_idx);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axis);

    const size_t extended_rank = target_shape_v.size() + 1;
    std::vector<int32_t> shape_vec(extended_rank, 1);
    shape_vec[num_heads_idx] = static_cast<int32_t>(num_groups);
    auto broadcast_shape = ov::op::v0::Constant::create(ov::element::i32, {extended_rank}, shape_vec);
    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);

    auto target_shape = ov::op::v0::Constant::create(ov::element::i32, {4}, target_shape_v);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(broadcast, target_shape, true);

    return reshape;
}

inline std::shared_ptr<ov::Node> make_qkv_transpose(ov::Output<ov::Node> qkv, std::vector<int64_t> order) {
    auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, {order.size()}, order);
    return std::make_shared<ov::op::v1::Transpose>(qkv, transpose_const);
}

inline std::shared_ptr<ov::Node> make_kv_rearrange(ov::Output<ov::Node> kv_past, ov::Output<ov::Node> beam_idx, int axis_val = 0) {
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, axis_val);
    return std::make_shared<ov::op::v8::Gather>(kv_past, beam_idx, axis, 0);
}

inline std::shared_ptr<ov::Model> make_llm_kv_cache_pattern(ov::Dimension batch = ov::Dimension::dynamic(),
                                                            ov::Dimension n_heads = ov::Dimension::dynamic(),
                                                            ov::Dimension n_features = ov::Dimension::dynamic(),
                                                            ov::element::Type_t element_type = ov::element::f32,
                                                            int64_t concat_axis = 2,
                                                            bool stateful = false,
                                                            bool fuse_cache_reorder = false,
                                                            bool build_state_initializer = false,
                                                            size_t num_groups = 1) {
    ov::PartialShape kv_cache_size = {batch, n_heads / num_groups, -1, n_features};
    ov::PartialShape new_token_size = {batch, -1, n_heads / num_groups, n_features};
    ov::PartialShape matmul_in_size = {batch, n_heads, -1, -1};

    auto in_kv_prev = std::make_shared<ov::op::v0::Parameter>(element_type, kv_cache_size);
    in_kv_prev->set_friendly_name("past_key_values");
    auto in_new_token = std::make_shared<ov::op::v0::Parameter>(element_type, new_token_size);
    in_new_token->set_friendly_name("new_token_input");
    auto in_matmul = std::make_shared<ov::op::v0::Parameter>(element_type, matmul_in_size);
    in_matmul->set_friendly_name("in_matmul");

    ov::ParameterVector params{in_kv_prev, in_new_token, in_matmul};
    std::shared_ptr<ov::Node> concat_input = in_kv_prev;
    if (fuse_cache_reorder) {
        auto in_beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
        in_beam_idx->set_friendly_name("beam_idx");
        params.push_back(in_beam_idx);
        concat_input = make_kv_rearrange(in_kv_prev, in_beam_idx);
    }

    auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, {new_token_size.size()}, {0, 2, 1, 3});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(in_new_token, transpose_const);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{concat_input, transpose}, concat_axis);
    auto convert = std::make_shared<ov::op::v0::Convert>(concat, element_type);
    auto kv_present = std::make_shared<ov::op::v0::Result>(convert);
    kv_present->set_friendly_name("present_key_values");

    std::shared_ptr<ov::Node> kv_input = concat;
    if (num_groups > 1) {
        auto nh = static_cast<int32_t>(n_heads.get_length());
        auto hs = static_cast<int32_t>(n_features.get_length());
        std::vector<int32_t> target_shape = { 0, nh, -1, hs };
        kv_input = make_gqa(kv_input, num_groups, target_shape, nh);
    }

    auto matmul = std::make_shared<ov::op::v0::MatMul>(in_matmul, kv_input, false, false);
    auto matmul_out = std::make_shared<ov::op::v0::Result>(matmul);
    matmul_out->set_friendly_name("matmul_out");

    ov::ResultVector results{kv_present, matmul_out};
    auto model = std::make_shared<ov::Model>(results, params, "LLM-KV-Cache");
    if (stateful) {
        ov::pass::MakeStateful({{in_kv_prev, kv_present}}).run_on_model(model);
        if (build_state_initializer) {
            auto state_initializer = make_state_initializer(in_new_token, element_type, kv_cache_size, {0, 1, 2, 3});
            for (auto op : model->get_ops()) {
                if (auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
                    read_value->set_arguments(ov::OutputVector{state_initializer});
                }
            }
        }
    }

    model->validate_nodes_and_infer_types();

    return model;
}

inline std::shared_ptr<ov::Model> make_llm_kv_cache_sdpa_pattern(ov::Dimension batch = ov::Dimension::dynamic(),
                                                                 ov::Dimension n_heads = ov::Dimension::dynamic(),
                                                                 ov::Dimension n_features = ov::Dimension::dynamic(),
                                                                 ov::element::Type_t element_type = ov::element::f32,
                                                                 std::vector<int64_t> qkv_order = {0, 1, 2, 3},
                                                                 bool causal = false,
                                                                 bool with_mask = false,
                                                                 bool with_scale = false,
                                                                 bool stateful = false,
                                                                 bool fuse_cache_reorder = false,
                                                                 size_t num_groups = 1) {
    ov::PartialShape kv_cache_size_def = {batch, n_heads / num_groups, -1, n_features};
    ov::PartialShape new_token_size_def = {batch, n_heads / num_groups, -1, n_features};
    ov::PartialShape q_size_def = {batch, n_heads, -1, n_features};

    ov::PartialShape kv_cache_size = ov::PartialShape::dynamic(4);
    ov::PartialShape new_token_size = ov::PartialShape::dynamic(4);
    ov::PartialShape q_size = ov::PartialShape::dynamic(4);

    for (size_t i = 0; i < kv_cache_size_def.size(); i++) {
        kv_cache_size[qkv_order[i]] = kv_cache_size_def[i];
        new_token_size[qkv_order[i]] = new_token_size_def[i];
        q_size[qkv_order[i]] = q_size_def[i];
    }

    int64_t concat_axis = qkv_order[2];

    auto past_k = std::make_shared<ov::op::v0::Parameter>(element_type, kv_cache_size);
    past_k->set_friendly_name("past_k");

    auto past_v = std::make_shared<ov::op::v0::Parameter>(element_type, kv_cache_size);
    past_v->set_friendly_name("past_v");

    auto in_k_token = std::make_shared<ov::op::v0::Parameter>(element_type, new_token_size);
    in_k_token->set_friendly_name("new_k_token");

    auto in_v_token = std::make_shared<ov::op::v0::Parameter>(element_type, new_token_size);
    in_v_token->set_friendly_name("new_v_token");

    auto in_q = std::make_shared<ov::op::v0::Parameter>(element_type, q_size);
    in_q->set_friendly_name("in_q");

    ov::ParameterVector params{in_q, past_k, past_v, in_k_token, in_v_token };
    std::shared_ptr<ov::Node> concat_k_input = past_k;
    std::shared_ptr<ov::Node> concat_v_input = past_v;
    if (fuse_cache_reorder) {
        auto in_beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
        in_beam_idx->set_friendly_name("beam_idx");
        params.push_back(in_beam_idx);

        concat_k_input = make_kv_rearrange(past_k, in_beam_idx, qkv_order[0]);
        concat_v_input = make_kv_rearrange(past_v, in_beam_idx, qkv_order[0]);
    }

    auto concat_k = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{concat_k_input, in_k_token}, concat_axis);
    auto convert_k = std::make_shared<ov::op::v0::Convert>(concat_k, element_type);

    auto concat_v = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{concat_v_input, in_v_token}, concat_axis);
    auto convert_v = std::make_shared<ov::op::v0::Convert>(concat_v, element_type);

    auto present_k = std::make_shared<ov::op::v0::Result>(convert_k);
    present_k->set_friendly_name("present_k");

    auto present_v = std::make_shared<ov::op::v0::Result>(convert_v);
    present_v->set_friendly_name("present_v");

    std::shared_ptr<ov::Node> q = in_q;
    std::shared_ptr<ov::Node> k = concat_k;
    std::shared_ptr<ov::Node> v = concat_v;
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;

    if (with_mask) {
        mask = make_attention_mask(q, k, element_type, qkv_order);
    }

    if (with_mask && with_scale) {
        scale = ov::op::v0::Constant::create(element_type, ov::Shape{}, { 1.0f / std::sqrt(n_features.get_max_length())});
    }

    if (num_groups > 1) {
        auto nh = static_cast<int32_t>(n_heads.get_length());
        auto hs = static_cast<int32_t>(n_features.get_length());
        std::vector<int32_t> target_shape = { 0, nh, -1, hs };
        std::vector<int32_t> target_shape_transposed(target_shape.size());

        for (size_t i = 0; i < target_shape.size(); i++) {
            target_shape_transposed[qkv_order[i]] = target_shape[i];
        }

        k = make_gqa(k, num_groups, target_shape_transposed, nh);
        v = make_gqa(v, num_groups, target_shape_transposed, nh);
    }

    if (qkv_order != std::vector<int64_t>{0, 1, 2, 3}) {
        q = make_qkv_transpose(q, qkv_order);
        k = make_qkv_transpose(k, qkv_order);
        v = make_qkv_transpose(v, qkv_order);
    }

    std::shared_ptr<ov::Node> sdpa = nullptr;
    if (mask && scale) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, scale, causal);
    } else if (mask) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, causal);
    } else {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, causal);
    }
    sdpa->set_friendly_name("sdpa");

    auto sdpa_out = std::make_shared<ov::op::v0::Result>(sdpa);
    sdpa_out->set_friendly_name("sdpa_out");

    ov::ResultVector results{sdpa_out, present_k, present_v};
    auto model = std::make_shared<ov::Model>(results, params, "LLM-KV-Cache-SDPA");

    if (stateful) {
        ov::pass::MakeStateful({{past_k, present_k}, {past_v, present_v}}).run_on_model(model);
        auto state_initializer = make_state_initializer(in_v_token, element_type, kv_cache_size, qkv_order);
        for (auto op : model->get_ops()) {
            if (auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
                read_value->set_arguments(ov::OutputVector{state_initializer});
            }
        }
    }

    model->validate_nodes_and_infer_types();

    return model;
}

} // namespace tests
