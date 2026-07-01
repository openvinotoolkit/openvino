// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "model_builder_types.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace npuw {

struct SwiGLU {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;

    SwiGLU(size_t hs, size_t is, ov::element::Type prec, WeightFn wf)
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

struct GELU {
    size_t hidden_size;
    size_t intermediate_size;
    ov::element::Type precision;
    WeightFn weight_fn;
    WeightFn bias_fn;

    GELU(size_t hs, size_t is, ov::element::Type prec, WeightFn wf, WeightFn bf = {})
        : hidden_size(hs),
          intermediate_size(is),
          precision(prec),
          weight_fn(std::move(wf)),
          bias_fn(std::move(bf)) {}

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;
};

/// Builds a MoE FFN from the params build_llm resolves out of LLMConfig, keeping the config the
/// single source of truth. Empty = GPT-OSS MoEFFN; assign make_qwen3_moe_ffn for Qwen3.
using MoEFactoryFn = std::function<FFNFn(size_t hidden_size,
                                         size_t intermediate_size,
                                         size_t num_experts,
                                         size_t num_experts_per_tok,
                                         ov::element::Type precision)>;

/// GPT-OSS style batched MoE FFN matching NPUW's GPTOSSExpert + GPTOSSRouter patterns.
/// All experts compute on all tokens via Tile + 3D batched MatMul (no NonZero): fused
/// gate_up MatMul with Slice/Minimum/Swish + Clamp branches and a TopK→Softmax router.
/// Weight function must produce a Multiply→Convert→MatMul chain (default: i4 CompressedWeight)
/// for the isolation patterns to match. Shared constants across layers enable repeating
/// block detection. Conforms to FFNFn for drop-in use in transformer layer templates.
///
/// New topologies: add a functor + a MoEFactoryFn wrapper (see make_qwen3_moe_ffn), no central dispatch.
struct MoEFFN {
    size_t hidden_size, intermediate_size, num_experts, num_experts_per_tok;
    ov::element::Type precision;
    WeightFn weight_fn;

    /// Default weight_fn: CompressedWeight{i4, 0, SYMM_NO_ZP}.
    MoEFFN(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec, WeightFn wf = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;

private:
    // Shared across layers for matchRepeatedSubgraphs (created once in ctor)
    std::shared_ptr<ov::Node> tile_repeats, topk_k_const;
    std::shared_ptr<ov::Node> slice_step, slice_axis2, slice_start_0, slice_stop_is, slice_start_is, slice_stop_2is;
    std::shared_ptr<ov::Node> min_const, swish_beta, clamp_add_zero;
    std::shared_ptr<ov::Node> sl_start, sl_step_r, sl_axes, scatter_axis;
    std::shared_ptr<ov::Node> tp_order, unsq_axis, reduce_axis;
};

/// Qwen3 style batched MoE FFN matching NPUW's Qwen3Expert + Qwen3Router patterns
/// (real Qwen3-30B-A3B). Differs from MoEFFN in two ways: the expert uses separate gate/up
/// MatMuls (SwiGLU = Swish(gate) * up, single-input Swish) instead of a fused gate_up with
/// Clamp/Minimum branches, and the router is Softmax→TopK with an explicit ReduceSum→Divide
/// renormalization over the K selected experts (vs GPT-OSS's TopK→Softmax). The expert weight
/// chain is the same Multiply→Convert→MatMul the matchers bind to. Note: the shipped model
/// fuses gate_up via VariadicSplit; the matcher (and this builder) use the separate gate/up
/// form the partitioning pass keys on. Conforms to FFNFn; select via
/// LLMConfig::moe_factory = make_qwen3_moe_ffn (build_llm builds it from the config's MoE fields).
struct Qwen3MoEFFN {
    size_t hidden_size, intermediate_size, num_experts, num_experts_per_tok;
    ov::element::Type precision;
    WeightFn weight_fn;

    /// Default weight_fn: CompressedWeight{i4, 0, SYMM_NO_ZP}.
    Qwen3MoEFFN(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec, WeightFn wf = {});

    ov::Output<ov::Node> operator()(const ov::Output<ov::Node>& input, const std::string& name) const;

private:
    // Shared across layers for matchRepeatedSubgraphs (created once in ctor)
    std::shared_ptr<ov::Node> tile_repeats, topk_k_const, scatter_axis, tp_order, unsq_axis;
    std::shared_ptr<ov::Node> reduce_axis;    ///< final expert reduction (over experts, axis 0)
    std::shared_ptr<ov::Node> reduce_axis_k;  ///< router renormalization axis (over K)
};

/// MoEFactoryFn for the GPT-OSS topology — build_llm's default when LLMConfig::moe_factory is empty.
inline FFNFn make_gptoss_moe_ffn(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec) {
    return MoEFFN(hs, is, ne, k, prec);
}

/// MoEFactoryFn for the Qwen3 topology. Assign to LLMConfig::moe_factory to select it.
inline FFNFn make_qwen3_moe_ffn(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec) {
    return Qwen3MoEFFN(hs, is, ne, k, prec);
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
