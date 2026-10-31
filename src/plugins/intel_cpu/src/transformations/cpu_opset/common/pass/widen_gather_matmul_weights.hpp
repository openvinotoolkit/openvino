// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

// ============================================================================
// WidenGatherMatmulWeights
// ============================================================================
//
// Widens a GatherMatmul's compressed weight Constant to the narrowest element type the node
// actually supports, when its own type is not supported but can be represented losslessly by
// one that is.
//
// GatherMatmul accepts fewer compressed weight types than FullyConnected does (see
// GatherMatmul::getSupportedCompressedWeightsTypes) -- notably it has no u2 executor, while
// FullyConnected does. Without this pass a u2 weight tensor is simply not matched by
// ConvertGatherMatmulToGatherMatmulCompressed, so its Convert -> Subtract -> Multiply
// dequantization subgraph stays in the graph and constant folding materializes the weights in
// f32: a 16x expansion off a 2-bit type, which on a wide MoE model is many gigabytes. Paying
// 2x to reach a supported type is far cheaper than falling off the compressed path entirely.
//
// The widening is lossless (u2's [0..3] fits a nibble) and leaves the dequantization arithmetic
// untouched -- only the storage type of the weight Constant changes -- so the following
// compression pass matches and numerics are unaffected. Must run BEFORE
// ConvertGatherMatmulToGatherMatmulCompressed.
//
// This is a workaround for a missing executor, not a desirable state: when GatherMatmul gains
// native support for a type listed in kWidenings below, drop that entry.
class WidenGatherMatmulWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("WidenGatherMatmulWeights");

    // `supported_weights_types` is the node's own capability list, i.e.
    // GatherMatmul::getSupportedCompressedWeightsTypes(). A widening is applied only if its
    // source type is absent from that list and its target type is present, so the pass is a
    // no-op on a build/architecture where the narrow type is already supported.
    explicit WidenGatherMatmulWeights(const std::vector<ov::element::Type>& supported_weights_types);
};

}  // namespace ov::intel_cpu
