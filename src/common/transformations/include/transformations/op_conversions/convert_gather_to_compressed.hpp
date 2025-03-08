// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API CompressedGatherTransformation;
class TRANSFORMATIONS_API ConvertGatherToGatherCompressed;
class TRANSFORMATIONS_API MoveDecompressionAfterGather;

}  // namespace pass
}  // namespace ov

/*
 * ConvertGatherToGatherCompressed transform gather node with constant weight decompression pattern(U8/NF4/U4/I4 +
 * Subtract + Multiply) to GatherCompressed node, which handle decompression internally.
 *
 *                        Subtract_const(U8/NF4/U4/I4)
 *                             /
 *    Weights(U8/NF4/U4/I4)  Convert(F32)                              Weights  Subtract_const Multiply_const Indices
 *       |                 /                                     (U8/NF4/U4/I4) (U8/NF4/U4/I4) (F32)          (I32)
 *    Convert(F32)   Reshape(optional)                                  \            \        /            /
 *            \        /       Multiply_const(F32)      ------>           \           \      /           /
 *            Subtract(optional)     /                                      \          \    /          /
 *                  \       Reshape(optional)                                 \         \  /         /
 *                   \       /                                                    GatherCompressed
 *    Indices(I32)    Multiply
 *            \     /
 *             Gather
 */
class ov::pass::ConvertGatherToGatherCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGatherToGatherCompressed");
    ConvertGatherToGatherCompressed();
};

/*
 * MoveDecompressionAfterGather transform gather node with constant weight decompression pattern(FP16/BF16 +
 * convert(FP32)) to gather node with compressed(FP16/BF16) weight, and move decompression after gather node.
 *
 *    Weights(FP16/BF16)                            Weights(FP16/BF16) Indices(I32)
 *         |                                                   \         /
 *    Convert(F32)   Indices(I32)    ------>                 Gather(FP16/BF16)
 *          \           /                                           |
 *           Gather(F32)                                        Convert(F32)
 *
 */
class ov::pass::MoveDecompressionAfterGather : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveDecompressionAfterGather");
    MoveDecompressionAfterGather();
};

class ov::pass::CompressedGatherTransformation : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("CompressedGatherTransformation");
    CompressedGatherTransformation() {
        add_matcher<ov::pass::ConvertGatherToGatherCompressed>();
        add_matcher<ov::pass::MoveDecompressionAfterGather>();
    }
};
