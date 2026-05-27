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

/// GPT-OSS style batched MoE FFN matching NPUW's GPTOSSExpert + GPTOSSRouter patterns.
/// All experts compute on all tokens via Tile + 3D batched MatMul (no NonZero).
/// Weight function must produce a Multiply→Convert→MatMul chain (default: i4 CompressedWeight)
/// for the isolation patterns to match. Shared constants across layers enable repeating
/// block detection. Conforms to FFNFn for drop-in use in transformer layer templates.
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

}  // namespace npuw
}  // namespace test
}  // namespace ov
