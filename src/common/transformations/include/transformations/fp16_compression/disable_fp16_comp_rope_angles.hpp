// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Keeps the RoPE angles computed at run time in f32 when the model is compressed to f16.
 *
 * Decoder and encoder exports compute the rotary cos/sin inputs at run time from integer positions. In f16 the angle
 * position * inv_freq is quantized (the f16 step is 2 at 2048 and 16 at 16384), so cos/sin lose all meaning beyond a
 * few thousand positions. MarkSugraphsToKeepInMixedPrecision, run inside ConvertPrecision, does not reach these
 * angles: the positions are a Parameter or an integer Range, and none of the nodes below is one of its seeds.
 *
 * Before:
 *
 *   int positions -> Convert(f32) -> [Broadcast] -> [Reshape|Unsqueeze] -+
 *                                                                        +-> MatMul|Multiply -> [Reshape] -> ...
 *                                                       any f32 freq  ---+
 *   ... -> [Transpose] -> [Concat(x, x)] -+-> Sin -> consumers
 *                                         +-> Cos -> consumers
 *
 * After: the nodes from the position Convert to Sin/Cos are marked with disable_conversion(f16), and a Convert(f32)
 * is inserted between each of Sin/Cos and its consumers:
 *
 *   ... -+-> Sin -> Convert(f32) -> consumers
 *        +-> Cos -> Convert(f32) -> consumers
 *
 * The inserted Convert stops the downward propagation of the marks, so the rotation of q/k, the fused RoPE and SDPA
 * keep the inference precision. As it does not change the type, the upward propagation of the marks goes through it:
 * a consumer that ConvertPrecision keeps in f32 (e.g. 1 / Maximum(1 - cos, eps)) reads the f32 cos/sin. Inside
 * ConvertPrecision, AlignMixedFP32FP16Types puts an f32 -> f16 Convert between Sin/Cos and the other consumers, and
 * the inserted Convert, a no-op, is removed. The upward propagation of the marks still reaches a non-constant
 * frequency input (e.g. LongRoPE scaling factors).
 *
 * Applied only when:
 *  - the position Convert converts to f32 the output of a non-Constant node of integer type;
 *  - the frequencies and the Sin/Cos are f32;
 *  - the Concat, if any, concatenates the same angles twice;
 *  - the angles (or their Concat) feed both a Sin and a Cos;
 *  - every node of the chain, apart from Sin/Cos, has only consumers that belong to a matched chain (or ShapeOf
 *    consumers, for the nodes before the MatMul|Multiply); a chain that shares nodes with a rejected chain is rejected
 *    too;
 *  - the transformation callback does not reject the Sin/Cos.
 * The bodies of If, Loop and TensorIterator that allow transformations are processed as separate models: a chain is
 * matched only when it lies entirely in one body. Positions converted to f32 outside of a body, or angles computed
 * outside of the body of their Sin/Cos, are not matched. A frequency input that enters a body as a body Parameter is
 * converted to f16 with the body inputs by ConvertPrecision, so only its product with the positions is kept in f32.
 * In a body, ConvertPrecision does not remove the inserted Convert: it stays (f16 -> f16, or f32 -> f32 when a
 * precision sensitive consumer marks it) until EliminateConvert.
 * No Convert is inserted in front of a Result, a Convert or a consumer already kept in f32.
 */
class TRANSFORMATIONS_API DisableFP16CompForRopeAngles : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("DisableFP16CompForRopeAngles");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
