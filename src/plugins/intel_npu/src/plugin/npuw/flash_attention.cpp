// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "flash_attention.hpp"
#include "pyramid_attention.hpp"



#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"



namespace ov::npuw::function {

    using Ptr = std::shared_ptr<Node>;

ov::OutputVector make_hfa_tile(
    Ptr q, Ptr k, Ptr v, Ptr m,
    Ptr past_max, Ptr past_d, Ptr past_acc) {
    // qk = op.matmul(q, k, False, True)
    // qkm = op.add(qk, m)
    // maxx = op.maximum(past_max, op.reduce_max(qkm, -1, True))
    // p = op.exp(op.subtract(qkm, maxx))
    // l = op.reduce_sum(p, -1, True)
    // alpha = op.exp(op.subtract(past_max, maxx))
    // d = op.add(op.multiply(past_d, alpha), l)
    // acc = op.add(op.multiply(past_acc, alpha), op.matmul(p, v, False, True))
    // return acc, maxx, d

    auto qk = std::make_shared<ov::op::v0::MatMul>(q, k, false, true);
    auto qkm = std::make_shared<ov::op::v1::Add>(qk, m);
    auto axis_minus_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axis_minus_1, true);
    auto maxx = std::make_shared<ov::op::v1::Maximum>(past_max, reduce_max);
    auto sub1 = std::make_shared<ov::op::v1::Subtract>(qkm, maxx);
    auto p  = std::make_shared<ov::op::v0::Exp>(sub1);
    auto l  = std::make_shared<ov::op::v1::ReduceSum>(p, axis_minus_1, true);
    auto sub2 = std::make_shared<ov::op::v1::Subtract>(past_max, maxx);
    auto alpha = std::make_shared<ov::op::v0::Exp>(sub2);
    auto mul1 = std::make_shared<ov::op::v1::Multiply>(past_d, alpha);
    auto d = std::make_shared<ov::op::v1::Add>(mul1, l);
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(past_acc, alpha);
    auto matmul2 = std::make_shared<ov::op::v0::MatMul>(p, v, false, true);
    auto acc = std::make_shared<ov::op::v1::Add>(mul2, matmul2);
    return {acc, maxx, d};
}


Ptr make_concat_with_past(Ptr input_past_key, Ptr input_past_value, int concat_axis) {
    auto cnt = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{input_past_key, input_past_value}, int64_t(concat_axis));
    auto axis_2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto unsq = std::make_shared<ov::op::v0::Unsqueeze>(cnt, axis_2);

    auto s = unsq->get_shape();
    auto bcast_shape = ov::op::v0::Constant::create(
        ov::element::i64,
        {5},
        std::vector<size_t>{s[0], s[1], 4, s[3], s[4]}); // 4 is hardcoded for now

    auto bcast = std::make_shared<ov::op::v3::Broadcast>(unsq, bcast_shape);

    auto new_shape = ov::op::v0::Constant::create(
        ov::element::i64,
        {4},
        std::vector<size_t>{s[0], s[1] * 4, s[3], s[4]});

    auto reshaped = std::make_shared<ov::op::v1::Reshape>(bcast, new_shape, true);

    return reshaped;
    // def make_full_k(self):
    // ii = self.input_tensors
    // full_k = np.concatenate([ii[Inputs.PAST_K.value], ii[Inputs.K.value]], axis=-2)
    // full_k = np.expand_dims(full_k, axis=2)
    // full_k = np.tile(full_k, [1,1,4,1,1]) # FIXME: Hardcoded shapes
    // full_k = np.reshape(full_k, [1,32,8192,128])
    // return full_k
}

// make_full_v_concat {

// }
ov::ParameterVector find_params_for_node(const OutputVector& output) {
    ov::ParameterVector params;
    std::stack<std::shared_ptr<ov::Node>> nodes_stack;
    std::stack<int> s;
    for (auto &output_element : output)
    nodes_stack.push(output_element.get_node_shared_ptr());

    std::unordered_set<ov::Node*> visited;

    while (!nodes_stack.empty()) {
        auto node = nodes_stack.top();
        nodes_stack.pop();

        if (!visited.insert(node.get()).second)
            continue;

        if (auto p = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            params.push_back(p);
            continue;
        }

        for (auto& input : node->inputs()) {
            auto src = input.get_source_output().get_node_shared_ptr();
            if (src)
                nodes_stack.push(src);
        }
    }

    return params;
}

std::optional<FlashAttention> FlashAttention::from(const std::shared_ptr<ov::Model>& model) {
    // Find SDPA pattern nodes in the model
    auto pattern_nodes = find_sdpa_pattern_nodes(model);
    if (!pattern_nodes.is_valid()) {
        LOG_WARN("Could not find SDPA pattern in model");
        return std::nullopt;
    }

    LOG_INFO("SDPA_located for flash-attention: matmul=" << pattern_nodes.matmul1_node->get_friendly_name());

     // copy from Pyramid attention
     // Extract query_length and full_context_length from Softmax output shape
     auto softmax_output_shape = pattern_nodes.softmax_node->get_output_shape(0);
     size_t query_length = 0;
     size_t past_kv_length = 0;
     size_t full_context_length = 0;

     if (softmax_output_shape.size() >= 2) {
         full_context_length = softmax_output_shape.back();                     // Last dimension
         query_length = softmax_output_shape[softmax_output_shape.size() - 2];  // Second-to-last dimension

         LOG_DEBUG("Extracted from Softmax output shape:");
         LOG_DEBUG("  Query length: " << query_length);
         LOG_DEBUG("  Full context length: " << full_context_length);
     } else {
         LOG_WARN("Softmax output shape has insufficient dimensions: " << softmax_output_shape.size());
         return std::nullopt;
     }

     // Early return for invalid parameters
     if (query_length == 0 || full_context_length == 0 || full_context_length < query_length) {
         LOG_WARN("Invalid query_length (" << query_length << ") or full_context_length (" << full_context_length
                                           << ") for flash attention");
         return std::nullopt;
     }


    FlashAttention hfa;
    hfa._full_context_length = full_context_length;
    hfa._query_length = query_length;

    // // A bad test but it is what it is
    // // Find the attention inputs with dynamic range
    // const auto& f_params = model->get_parameters();
    // NPUW_ASSERT(f_params.size() > 0);
    // for (auto&& param : f_params) {
    //     hfa._inputs.push_back(ov::npuw::function::FlashAttention::Param{param, dim_idx});
    // }


    // # FIXME: For a single-tile debug, K & V are pre-concatenated
    auto full_k = pattern_nodes.matmul1_node->input(1);
    auto full_v = pattern_nodes.matmul2_node->input(1);

    // ii = self.input_tensors
    // full_k = self.make_full_k()
    //auto full_k_shape = full_k.get_shape();
    // full_v = self.make_full_v()
    //auto full_v_shape = full_v.get_shape();
    // past_a = np.zeros([1,32,1024,128], np.half)
    auto past_a_shape = pattern_nodes.matmul2_node->output(0).get_shape();
    // past_m = np.ones ([1,32,1024,1], np.half)*(np.float16(-65500.0))
    auto past_m_shape = pattern_nodes.matmul2_node->output(0).get_shape();
    past_m_shape[3] = 1;
    // past_d = np.zeros([1,32,1024,1], np.half)
    auto past_d_shape = pattern_nodes.matmul2_node->output(0).get_shape();
    past_d_shape[3] = 1;

    // in_q = op.parameter(ii[Inputs.Q.value].shape, ov.Type.f16)
    auto in_q_shape = pattern_nodes.matmul1_node->input(0).get_shape();

    // TODO: find this spatial dimension idx and provide constant for tile size

    // in_m = op.parameter([1,1,1024,TSZ], ov.Type.f16)
    auto this_m_shape = pattern_nodes.add_node->input(1).get_shape();
    this_m_shape[3] = 1024;
    // in_k = op.parameter([1,32,TSZ,128], ov.Type.f16)
    auto this_k_shape = full_k.get_shape();
    this_k_shape[2] = 1024;
    // in_v = op.parameter([1,32,128,TSZ], ov.Type.f16)
    auto this_v_shape = full_v.get_shape();
    this_v_shape[3] = 1024;



    auto in_this_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, this_k_shape);
    auto in_this_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, this_v_shape);
    auto in_past_m = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_m_shape);
    auto in_past_d = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_d_shape);
    auto in_past_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_a_shape);

    auto in_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, in_q_shape);
    auto in_this_m = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, this_m_shape);

    LOG_INFO("HFA tile shapes:" <<
        in_q <<  in_this_k <<  in_this_v << in_this_m << in_past_m << in_past_d << in_past_a);
    //acc, maxx, d = self.ov_hfa_tile(in_q, in_k, in_v, in_m, in_past_m, in_past_d, in_past_a)

    auto tile_result = make_hfa_tile(in_q, in_this_k, in_this_v, in_this_m, in_past_m, in_past_d, in_past_a);
    hfa.models.resize(FlashAttention::eLast);
    hfa.models[FlashAttention::eTile] = std::make_shared<ov::Model>(
        tile_result,
        ov::ParameterVector{in_past_a, in_past_m, in_past_d, in_this_k, in_this_v, in_q, in_this_m},
        "hfa_tile");


    // TODO: ES for some reason this doesnt work always - some unreferenced parameters double check
    // auto kv_cache_concat_results = ov::OutputVector{
    //     full_k.get_source_output().get_node_shared_ptr(),
    //     full_v.get_source_output().get_node_shared_ptr()};

    // hfa.models[FlashAttention::eConcat] = std::make_shared<ov::Model>(
    //     ov::as_result_vector(kv_cache_concat_results),
    //     find_params_for_node(kv_cache_concat_results),
    //    "hfa_kv_concat");

    auto past_k = find_params_for_node({full_k.get_source_output().get_node_shared_ptr()});
    auto in_pask_kk = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_k[0]->get_shape());
    in_pask_kk->set_friendly_name(past_k[0]->get_friendly_name());
    auto in_pask_kv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_k[1]->get_shape());

    auto past_v = find_params_for_node({full_v.get_source_output().get_node_shared_ptr()});
    auto in_pask_vk = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_v[0]->get_shape());
    in_pask_vk->set_friendly_name(past_v[0]->get_friendly_name());
    auto in_pask_vv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_v[1]->get_shape());
    auto concat1_sub = make_concat_with_past(in_pask_kk, in_pask_kv, 2);
    auto concat2_sub = make_concat_with_past(in_pask_vk, in_pask_vv, 3);

    hfa.models[FlashAttention::eConcat] = std::make_shared<ov::Model>(
         ov::as_result_vector({concat1_sub, concat2_sub}),
         ov::ParameterVector{in_pask_kk, in_pask_kv, in_pask_vk, in_pask_vv},
        "hfa_kv_concat");

    // final = past_a / past_d
    // final = np.transpose(final, axes=(0,2,1,3))
    // final = np.reshape(final, [1,1024,4096])

    auto div1 = std::make_shared<ov::op::v1::Divide>(tile_result[0], tile_result[2], true);
    std::vector<size_t> order = {0, 2, 1, 3};
    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, {order.size()}, order);
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(div1, transpose_order);

    auto final_shape_pattern = ov::op::v0::Constant::create(
        ov::element::i64,
        {3},
        std::vector<int64_t>{0, 0, -1});

    auto reshaped1 = std::make_shared<ov::op::v1::Reshape>(transpose1, final_shape_pattern, true);
    auto result1 = std::make_shared<ov::op::v0::Result>(reshaped1);

    // can be also last model - hfa_tile + transpose + divide
    hfa.models[FlashAttention::eDivide] = std::make_shared<ov::Model>(
        ov::ResultVector{result1},
        ov::ParameterVector{in_past_a, in_past_m, in_past_d, in_this_k, in_this_v, in_q, in_this_m},
        "hfa_final_tile");

    return std::move(hfa);
}

}  // namespace ov::npuw::function

namespace ov::npuw::compiled {
FlashAttention:: FlashAttention(const function::FlashAttention& func_flash_attention)
    : _models_to_compile(func_flash_attention.models)
    , _query_length(func_flash_attention._query_length)
    , _full_context_length(func_flash_attention._full_context_length)
    , _num_tiles(func_flash_attention.num_tiles()) {
    LOG_INFO("Constructing compiled::FlashAttention ");

    // fixed params for attention
    // params.reserve();
    // // Extract metadata from each model
    // for (size_t i = 0; i < num_models; ++i) {
    //     const auto& func_attn = func_pyramid._attentions[i];
    //     const auto& model = func_pyramid._models[i];

    //     // Build attention info
    //     PyramidAttentionInfo attention_info;
    //     attention_info.params.reserve(func_attn._inputs.size());

    //     for (const auto& input : func_attn._inputs) {
    //         std::size_t p_idx = model->get_parameter_index(input.param);
    //         attention_info.params.push_back({p_idx, input.dim});
    //     }
}
}   // namespace ov::npuw::compiled


namespace ov::npuw::runtime::flash_attention {

PositionIDs::PositionIDs(std::size_t param_idx,
    const ov::npuw::compiled::FlashAttention& d,
    const ov::ISyncInferRequest& rq)
      : m_position_ids_idx(param_idx),
        m_flash_attention(d),
        m_rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    m_case = m_flash_attention.get()._query_length == 1 ? Case::GENERATE : Case::PREFILL;
}

void PositionIDs::prepare(int64_t past_len) {
    const auto& iport = m_rq.get_compiled_model()->inputs()[m_position_ids_idx];
    const auto in_tensor = m_rq.get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();
    const auto pos_ids_len = static_cast<int64_t>(in_dims.back());

    // There's several cases possible:
    // a. Prefill input_ids, including chunk
    // b. Generate input_ids, 1
    // c. Generate input_ids, N (speculative)
    // Prefill (even chunked) is left-padded, so for (a) it's enough to take the last element.
    // Same works for b (there's no choice).
    // c may require traversing the tensor backwards as Generate with N>1 is right_padded (?)

    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (auto idx = pos_ids_len - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            // Initialize fields
            m_current_length = pos_data_ptr[idx];
            switch (m_case) {
                case Case::GENERATE:
                    // decode case, we have pos_id-1 past elements to take from kvcache
                    m_past_length = m_current_length;
                    break;
                case Case::PREFILL: {
                    // chunked prefill case. calculate the past_length in full chunks
                    // FIXME: We know too much about chunking here
                    auto query_size = m_flash_attention.get()._query_length;
                    m_past_length = ((past_len + query_size - 1) / query_size) * query_size;
                    break;
                }
                default:
                    NPUW_ASSERT(false && "Reached the unreachable code");
            }
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    m_current_length = -1;
}
Selector::Ptr PositionIDs::find(const compiled::FlashAttention& flash, const ov::ISyncInferRequest& rq) {
    auto is_position_ids = [](const ov::Output<const ov::Node>& p) {
        const auto& shape = p.get_shape();
        // FIXME: 2D/3D position IDs are not supported here YET
        return p.get_node()->get_friendly_name() == "position_ids" &&
               (shape.size() == 1 || (shape.size() == 2 && shape[0] == 1));
    };

    const auto& inputs = rq.get_inputs();
    auto pos_ids_iter = std::find_if(inputs.begin(), inputs.end(), is_position_ids);
    if (pos_ids_iter != inputs.end()) {
        const auto param_idx = std::distance(inputs.begin(), pos_ids_iter);
        return Selector::Ptr{new PositionIDs(param_idx, flash, rq)};
    }
    return Selector::Ptr{};
}

int64_t PositionIDs::length() const {
    return m_current_length;
}

int64_t PositionIDs::past_length() const {
    return m_past_length;
}


}  // namespace ov::npuw::runtime::flash_attention