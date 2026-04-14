// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "concat_sdp_kv_bench.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov {
namespace test {

template <typename IT, typename T>
static void strided_iota_bench(IT first, size_t n, T value, T stride) {
    for (size_t i = 0; i < n; i++) {
        *first++ = value;
        value += stride;
    }
}

// GQA broadcast: [B, Hk, L, S] -> [B, H, L, S] via Unsqueeze+Broadcast+Reshape.
// Inserts the Unsqueeze(dim=2) + Broadcast({1,1,group_size,1,1}) + Reshape({0,0,H,S})
// pattern that the CPU plugin fuses into its internal SDPA GQA path.
static std::shared_ptr<ov::Node> make_gqa_broadcast(const std::shared_ptr<ov::Node>& kv_node,
                                                    const ov::PartialShape& q_shape) {
    // q_shape is [B, H, L, S], kv_node output is [B, Hk, L, S].
    auto H = q_shape[1].get_length();
    auto S = q_shape[3].get_length();
    auto Hk = kv_node->get_output_partial_shape(0)[1].get_length();
    auto group_size = H / Hk;

    // [B, Hk, L, S] -> [B, Hk, 1, L, S]
    auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {2});
    auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv_node, axis);

    // [B, Hk, 1, L, S] -> [B, Hk, group_size, L, S]
    auto target = ov::op::v0::Constant::create(ov::element::i32,
                                               {5},
                                               std::vector<int32_t>{1, 1, static_cast<int32_t>(group_size), 1, 1});
    auto broadcasted =
        std::make_shared<ov::op::v3::Broadcast>(unsqueezed, target, ov::op::BroadcastType::BIDIRECTIONAL);

    // [B, Hk, group_size, L, S] -> [B, H, L, S]  (merge dims 1,2)
    auto reshape_target =
        ov::op::v0::Constant::create(ov::element::i32,
                                     {4},
                                     std::vector<int32_t>{0, static_cast<int32_t>(H), -1, static_cast<int32_t>(S)});
    return std::make_shared<ov::op::v1::Reshape>(broadcasted, reshape_target, true);
}

std::string ConcatSDPKVBenchBase::getTestCaseName(const testing::TestParamInfo<ConcatSDPKVBenchParams>& obj) {
    const auto& [extraConfig, inputShapes, kCacheMode, vCacheMode, groupSize, rotationMode] = obj.param;
    std::ostringstream result;
    result << "IS=";
    for (const auto& shape : inputShapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << "TS=";
    for (const auto& shape : inputShapes) {
        result << "(";
        if (!shape.second.empty()) {
            for (const auto& itr : shape.second) {
                result << ov::test::utils::vec2str(itr);
            }
        }
        result << ")_";
    }
    for (const auto& [key, val] : extraConfig) {
        result << key << "=" << val.as<std::string>() << "_";
    }
    result << "K=" << kCacheMode << "_V=" << vCacheMode;
    if (groupSize > 0) {
        result << "_GS=" << groupSize;
    }
    result << "_Rot=" << rotationMode;
    return result.str();
}

void ConcatSDPKVBenchBase::SetUp() {
    const auto& [extraConfig, inputShapes, kCacheMode, vCacheMode, groupSize, rotationMode] = this->GetParam();
    targetDevice = ov::test::utils::DEVICE_CPU;
    configuration.insert(extraConfig.begin(), extraConfig.end());

    auto is_codec = [](const std::string& mode) {
        return mode == "tbq3" || mode == "tbq4" || mode == "tbq3_qjl" || mode == "tbq4_qjl" || mode == "polar3" ||
               mode == "polar4";
    };

    // Configure K cache.
    if (is_codec(kCacheMode)) {
        configuration["KEY_CACHE_CODEC"] = kCacheMode;
    } else if (kCacheMode == "u8" || kCacheMode == "u4") {
        configuration["KEY_CACHE_PRECISION"] = kCacheMode;
    }
    // "none" — no extra config for K.

    // Configure V cache.
    if (is_codec(vCacheMode)) {
        configuration["VALUE_CACHE_CODEC"] = vCacheMode;
    } else if (vCacheMode == "u8" || vCacheMode == "u4") {
        configuration["VALUE_CACHE_PRECISION"] = vCacheMode;
    }
    // "none" — no extra config for V.

    // Rotation mode is controlled via OV_TURBOQ_ROTATION env var at process start.

    // Configure group size for u8 (ignored by codecs).
    if (groupSize > 0) {
        configuration["KEY_CACHE_GROUP_SIZE"] = std::to_string(groupSize);
        configuration["VALUE_CACHE_GROUP_SIZE"] = std::to_string(groupSize);
    }

    // Enable profiling for per-node timing.
    configuration["PERF_COUNT"] = "YES";

    init_input_shapes(inputShapes);

    // Determine GQA configuration.
    // 2 shapes: [Q/K/V, past_KV] — H == Hk, no GQA.
    // 3 shapes: [Q, past_KV, new_KV] — H != Hk, GQA with broadcast.
    const bool is_gqa = inputDynamicShapes.size() > 2;
    const auto& q_shape = inputDynamicShapes[0];
    const auto& kv_past_shape = inputDynamicShapes[1];
    const auto& kv_new_shape = is_gqa ? inputDynamicShapes[2] : inputDynamicShapes[0];

    ov::ParameterVector inputParams;
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ElementType::f32, q_shape));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ElementType::f32, kv_new_shape));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ElementType::f32, kv_new_shape));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");

    // past K/V state
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ElementType::f32, kv_past_shape));
    auto var_k =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_past_shape, ElementType::f32, "pastk"});
    auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
    pastk->set_friendly_name("pastk_r");

    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ElementType::f32, kv_past_shape));
    auto var_v =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_past_shape, ElementType::f32, "pastv"});
    auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[4], var_v);
    pastv->set_friendly_name("pastv_r");

    // beam_idx
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    inputParams.push_back(beam_idx);

    auto gatherK =
        std::make_shared<ov::op::v8::Gather>(pastk, beam_idx, op::v0::Constant::create(ElementType::i32, {}, {0}));
    auto gatherV =
        std::make_shared<ov::op::v8::Gather>(pastv, beam_idx, op::v0::Constant::create(ElementType::i32, {}, {0}));
    auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, 2);
    auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, 2);

    // For GQA: broadcast [B, Hk, L, S] -> [B, H, L, S] via Unsqueeze+Broadcast+Reshape.
    // This is the pattern the CPU plugin fuses into its internal SDPA node.
    std::shared_ptr<ov::Node> sdpa_k = concatK;
    std::shared_ptr<ov::Node> sdpa_v = concatV;
    if (is_gqa) {
        sdpa_k = make_gqa_broadcast(concatK, q_shape);
        sdpa_v = make_gqa_broadcast(concatV, q_shape);
    }

    auto sdp = std::make_shared<ov::op::v13::ScaledDotProductAttention>(inputParams[0], sdpa_k, sdpa_v, false);
    sdp->set_friendly_name("mha");
    auto add = std::make_shared<ov::op::v1::Add>(sdp, op::v0::Constant::create(ElementType::f32, {1}, {1.0f}));
    auto pastk_assign = std::make_shared<op::v6::Assign>(concatK, var_k);
    auto pastv_assign = std::make_shared<op::v6::Assign>(concatV, var_v);
    pastk_assign->set_friendly_name("pastk_w");
    pastv_assign->set_friendly_name("pastv_w");

    ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    SinkVector sinks{pastk_assign, pastv_assign};
    function = std::make_shared<ov::Model>(results, sinks, inputParams, "ConcatSDPKVBench");
}

void ConcatSDPKVBenchBase::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& params = function->get_parameters();
    // params: q(0), k(1), v(2), pastk_init(3), pastv_init(4), beam_idx(5)
    // targetInputStaticShapes: [0]=Q, [1]=past_KV, [2]=new_KV (optional, else same as [0])
    const auto& kv_new_static =
        (targetInputStaticShapes.size() > 2) ? targetInputStaticShapes[2] : targetInputStaticShapes[0];
    // q
    {
        ov::Tensor t{params[0]->get_element_type(), targetInputStaticShapes[0]};
        strided_iota_bench(static_cast<float*>(t.data()), t.get_size(), 0.1f, 0.01f);
        inputs.insert({params[0], t});
    }
    // k, v
    for (size_t i = 1; i < 3; i++) {
        ov::Tensor t{params[i]->get_element_type(), kv_new_static};
        strided_iota_bench(static_cast<float*>(t.data()), t.get_size(), 0.1f * (i + 1), 0.01f);
        inputs.insert({params[i], t});
    }
    // past K/V init — empty (L0=0)
    for (size_t i = 3; i < 5; i++) {
        ov::Tensor t{params[i]->get_element_type(), targetInputStaticShapes[1]};
        strided_iota_bench(static_cast<float*>(t.data()), t.get_size(), 0.05f, 0.01f);
        inputs.insert({params[i], t});
    }
    // beam_idx — identity permutation [0, 1, ..., B-1]
    auto B = targetInputStaticShapes[0][0];
    ov::Tensor beam_t{ov::element::i32, {B}};
    auto* beam_data = static_cast<int32_t*>(beam_t.data());
    for (size_t b = 0; b < B; b++) {
        beam_data[b] = static_cast<int32_t>(b);
    }
    inputs.insert({params[5], beam_t});
}

}  // namespace test
}  // namespace ov
