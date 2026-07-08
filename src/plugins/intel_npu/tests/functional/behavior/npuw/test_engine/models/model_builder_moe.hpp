// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model_builder_types.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace npuw {

/// Builds a MoE FFN from the params build_llm resolves out of LLMConfig, keeping the config the
/// single source of truth. Empty = GPTOSSMoEFFN; assign make_qwen3_moe_ffn for Qwen3.
using MoEFactoryFn = std::function<FFNFn(size_t hidden_size,
                                         size_t intermediate_size,
                                         size_t num_experts,
                                         size_t num_experts_per_tok,
                                         ov::element::Type precision)>;

/// Scaffolding shared by the Tile-batched MoE FFN topologies (GPT-OSS, Qwen3, ...):
/// all experts compute on all tokens via Tile + 3D batched MatMul (no NonZero), and the
/// router scores are scattered to the full expert dimension, broadcast over the expert
/// outputs, and reduced back. Derived functors assemble operator() from these pieces;
/// only the router score computation and the expert body differ per topology.
///
/// Weight function must produce a Multiply→Convert→MatMul chain (default: i4 CompressedWeight)
/// for the isolation patterns to match. Shared constants across layers enable repeating
/// block detection. Derived types conform to FFNFn for drop-in use in transformer layer
/// templates. New topologies: derive a functor + add a MoEFactoryFn wrapper (see
/// make_qwen3_moe_ffn), no central dispatch.
struct BatchedMoEFFN {
    size_t hidden_size, intermediate_size, num_experts, num_experts_per_tok;
    ov::element::Type precision;
    WeightFn weight_fn;

protected:
    /// How reshape target shapes are emitted: concat_shape (GPT-OSS) or const_shape (Qwen3)
    /// — see those functions in model_builder_moe.cpp for why the form matters.
    using ShapeFn = std::shared_ptr<ov::Node> (*)(const std::vector<int32_t>&);

    /// Defaults weight_fn to CompressedWeight{i4, 0, SYMM_NO_ZP} when wf is empty.
    BatchedMoEFFN(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec, WeightFn wf, ShapeFn shape_fn);

    /// [B,S,H] input -> {".original_shape" ShapeOf, ".input_2d" [-1,H] Reshape}.
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> flatten_input(const ov::Output<ov::Node>& input,
                                                                        const std::string& name) const;

    /// Zero base for the router scatter: ShapeOf(logits) -> Broadcast(0).
    /// The ShapeOf friendly name differs per topology, so it is passed in full.
    ov::Output<ov::Node> router_zeros(const ov::Output<ov::Node>& logits,
                                      const std::string& shapeof_name,
                                      const std::string& zeros_name) const;

    /// Scattered per-expert scores [N,ne] -> [ne,1,N,1] for broadcasting over the expert
    /// outputs: Transpose -> Reshape -> Unsqueeze (the Qwen3Router/GPTOSSRouter tail).
    ov::Output<ov::Node> broadcast_router_scores(const ov::Output<ov::Node>& scattered,
                                                 const std::string& name) const;

    /// 2D input [N,H] -> all-expert batch [ne,N,H]: Tile -> Reshape.
    ov::Output<ov::Node> tile_to_experts(const ov::Output<ov::Node>& input_2d, const std::string& name) const;

    /// Expert outputs [ne,N,H] -> Reshape [ne,1,N,H] -> * router_scores -> ReduceSum over
    /// experts -> Reshape back to the original input shape. Returns the FFN output.
    ov::Output<ov::Node> weighted_expert_sum(const ov::Output<ov::Node>& expert_out_3d,
                                             const ov::Output<ov::Node>& router_scores,
                                             const ov::Output<ov::Node>& original_shape,
                                             const std::string& name) const;

    ShapeFn mk;
    // Shared across layers for matchRepeatedSubgraphs (created once in ctor)
    std::shared_ptr<ov::Node> tile_repeats, topk_k_const, scatter_axis, tp_order, unsq_axis, reduce_axis;
};

/// GPT-OSS style batched MoE FFN matching NPUW's GPTOSSExpert + GPTOSSRouter patterns:
/// fused gate_up MatMul with Slice/Minimum/Swish + Clamp branches and a TopK→Softmax router.
struct GPTOSSMoEFFN : BatchedMoEFFN {
    /// Default weight_fn: CompressedWeight{i4, 0, SYMM_NO_ZP}.
    GPTOSSMoEFFN(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec, WeightFn wf = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;

private:
    // GPT-OSS-specific constants: fused gate_up slicing, Clamp/Minimum, router Slice
    std::shared_ptr<ov::Node> slice_step, slice_axis2, slice_start_0, slice_stop_is, slice_start_is, slice_stop_2is;
    std::shared_ptr<ov::Node> min_const, swish_beta, clamp_add_zero;
    std::shared_ptr<ov::Node> sl_start, sl_step_r, sl_axes;
};

/// Qwen3 style batched MoE FFN matching NPUW's Qwen3Expert + Qwen3Router patterns
/// (real Qwen3-30B-A3B). Differs from GPTOSSMoEFFN in two ways: the expert uses separate gate/up
/// MatMuls (SwiGLU = Swish(gate) * up, single-input Swish) instead of a fused gate_up with
/// Clamp/Minimum branches, and the router is Softmax→TopK with an explicit ReduceSum→Divide
/// renormalization over the K selected experts (vs GPT-OSS's TopK→Softmax). The expert weight
/// chain is the same Multiply→Convert→MatMul the matchers bind to. Note: the shipped model
/// fuses gate_up via VariadicSplit; the matcher (and this builder) use the separate gate/up
/// form the partitioning pass keys on. Select via LLMConfig::moe_factory = make_qwen3_moe_ffn
/// (build_llm builds it from the config's MoE fields).
struct Qwen3MoEFFN : BatchedMoEFFN {
    /// Default weight_fn: CompressedWeight{i4, 0, SYMM_NO_ZP}.
    Qwen3MoEFFN(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec, WeightFn wf = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;

private:
    std::shared_ptr<ov::Node> reduce_axis_k;  ///< router renormalization axis (over K)
};

/// MoEFactoryFn for the GPT-OSS topology — build_llm's default when LLMConfig::moe_factory is empty.
inline FFNFn make_gptoss_moe_ffn(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec) {
    return GPTOSSMoEFFN(hs, is, ne, k, prec);
}

/// MoEFactoryFn for the Qwen3 topology. Assign to LLMConfig::moe_factory to select it.
inline FFNFn make_qwen3_moe_ffn(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec) {
    return Qwen3MoEFFN(hs, is, ne, k, prec);
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
