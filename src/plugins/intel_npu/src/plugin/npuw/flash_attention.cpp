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

    FlashAttention hfa;
    // TODO: where to get that
    hfa.nIterations = 8;

    // now we are creating loop/concat/denominator models
    // model->clone();


    // # FIXME: For a single-tile debug, K & V are pre-concatenated
    auto full_k = pattern_nodes.matmul1_node->input(1);
    auto full_v = pattern_nodes.matmul2_node->input(1);

    LOG_ERROR("debugging 104");
    // ii = self.input_tensors
    // full_k = self.make_full_k()
    auto full_k_shape = full_k.get_shape();
    // full_v = self.make_full_v()
    auto full_v_shape = full_v.get_shape();
    // past_a = np.zeros([1,32,1024,128], np.half)
    auto past_a_shape = pattern_nodes.matmul2_node->output(0).get_shape();
    // past_m = np.ones ([1,32,1024,1], np.half)*(np.float16(-65500.0))
    auto past_m_shape = pattern_nodes.matmul2_node->output(0).get_shape();
    past_m_shape[3] = 1;
    // past_d = np.zeros([1,32,1024,1], np.half)
    auto past_d_shape = pattern_nodes.matmul2_node->output(0).get_shape();
    past_m_shape[3] = 1;

    // in_q = op.parameter(ii[Inputs.Q.value].shape, ov.Type.f16)
    auto in_q_shape = pattern_nodes.matmul1_node->input(0).get_shape();
    // in_m = op.parameter(ii[Inputs.M.value].shape, ov.Type.f16)
    auto in_m_shape = pattern_nodes.add_node->input(1).get_shape();

    // acc, maxx, d = self.ov_hfa_tile(in_q, in_full_k, in_full_v, in_m,
    //                                 in_past_m, in_past_d, in_past_a)
    LOG_ERROR("debugging 103");
    auto in_full_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, full_k.get_shape());
    auto in_full_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, full_v.get_shape());
    auto in_past_m = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_m_shape);
    auto in_past_d = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_d_shape);
    auto in_past_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_a_shape);

    auto in_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, in_q_shape);
    auto in_m = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, in_m_shape);
    LOG_ERROR("debugging 102");

    auto tile_result = make_hfa_tile(in_q, in_full_k, in_full_v, in_m, in_past_m, in_past_d, in_past_a);
    hfa.models.resize(FlashAttention::eLast);
    hfa.models[FlashAttention::eTile] = std::make_shared<ov::Model>(
        tile_result,
        ov::ParameterVector{in_past_a, in_past_m, in_past_d, in_full_k, in_full_v, in_q, in_m},
        "hfa_tile");

    // TODO: ES for some reason this doesnt work always - some unreferenced parameters double check
    // auto kv_cache_concat_results = ov::OutputVector{
    //     full_k.get_source_output().get_node_shared_ptr(),
    //     full_v.get_source_output().get_node_shared_ptr()};

    // hfa.models[FlashAttention::eConcat] = std::make_shared<ov::Model>(
    //     ov::as_result_vector(kv_cache_concat_results),
    //     find_params_for_node(kv_cache_concat_results),
    //    "hfa_kv_concat");

    LOG_ERROR("debugging 101");
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
    LOG_ERROR("debugging 1");

    auto div1 = std::make_shared<ov::op::v1::Divide>(tile_result[0], tile_result[2], true);
    std::vector<size_t> order = {0, 2, 1, 3};
    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, {order.size()}, order);
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(div1, transpose_order);

    LOG_ERROR("debugging 2");

    auto final_shape_pattern = ov::op::v0::Constant::create(
        ov::element::i64,
        {3},
        std::vector<int64_t>{0, 0, -1});

    auto reshaped1 = std::make_shared<ov::op::v1::Reshape>(transpose1, final_shape_pattern, true);
    auto result1 = std::make_shared<ov::op::v0::Result>(reshaped1);
    LOG_ERROR("debugging 3");

    // can be also last model - hfa_tile + transpose + divide
    hfa.models[FlashAttention::eDivide] = std::make_shared<ov::Model>(
        ov::ResultVector{result1},
        ov::ParameterVector{in_past_a, in_past_m, in_past_d, in_full_k, in_full_v, in_q, in_m},
        "hfa_final_tile");
    LOG_ERROR("debugging passed");

    return std::move(hfa);
}

}  // namespace ov::npuw::function