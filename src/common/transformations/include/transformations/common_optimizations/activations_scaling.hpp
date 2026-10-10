// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ActivationsScaling;

namespace activations_scaling {

class TRANSFORMATIONS_API ScaleDownSingleLayer;
class TRANSFORMATIONS_API EliminateScalarMul;
class TRANSFORMATIONS_API MulShareTransformation;
class TRANSFORMATIONS_API MoveDownScalarMul;

}  // namespace activations_scaling
}  // namespace pass
}  // namespace ov

// ActivationsScaling makes activation values smaller to prevent overflow due to the limited range of FP16
// This feature is controlled by ov::hint::activations_scale_factor.
// For example, when this property is set as 16, activations are divided by 16.
// If ov::hint::activations_scale_factor is less than or equal to zero, it is disabled.

/**
 * @ingroup ov_transformation_common_api
 * @brief Keeps activations of a reduced-precision model in range by scaling them down around
 * Convolution/MatMul layers and removing the compensating scale-up at normalization layers.
 *
 * Linear layers can produce activations beyond the range of the scaled precision (e.g. above 65504
 * for f16), which turns into inf and then NaN at the following normalization. Since
 * Conv(x / s) = Conv(x) / s, the pass divides the input of every Conv/MatMul whose input already has
 * the scaled precision by the scale factor, multiplies the output back, and moves that scale-up
 * Multiply down the graph (through Add, residual connections and data movement ops) with LPT. A
 * normalization is invariant to the input scale, Norm(s * x) = Norm(x) with epsilon / s^2, so the
 * scale-up is dropped there and the subgraph between two normalizations runs scaled.
 *
 * Before:                              After:
 *
 *   residual    x                        residual          x
 *      │        │                           │              │
 *      │     MatMul                    Multiply(1/s)  Multiply(1/s)
 *      │        │                           │              │
 *      └──Add───┘                           │           MatMul
 *          │                                └─────Add──────┘
 *        Norm(eps)                                 │
 *                                            Norm(eps / s^2)
 *
 * The pass runs its own pipeline on a copy of the caller's PassConfig: the caller's callbacks apply to
 * the inner passes, while the LPT transformations disabled here stay disabled only inside the pass.
 * A scale factor <= 0 disables the pass.
 */
class ov::pass::ActivationsScaling : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ActivationsScaling");
    ActivationsScaling(float scale_factor, ov::element::Type scaled_prec);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    float m_scale_factor;
    ov::element::Type m_scaled_prec;
};

// Add scale_down and scale_up layers around Convolution and MatMul nodes
// Conv/MatMul
//    ==>
// Multiply(scale_down by scale_factor) --> Conv/MatMul --> Multiply(scale_up by scale_factor)
class ov::pass::activations_scaling::ScaleDownSingleLayer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ScaleDownSingleLayer", "0");
    ScaleDownSingleLayer(float scale_factor, ov::element::Type scaled_prec);
};

// Normalization and ShapeOf have the following property.
//
// Norm(input * const_a) = Norm(input)
//
// So, we can skip Multiply that is connected to Normalization and ShapeOf.
//
// input --> Multiply --> Normalization/ShapeOf
//   ==>
// input --> Normalization/ShapeOf
class ov::pass::activations_scaling::EliminateScalarMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateScalarMul", "0");
    EliminateScalarMul();
};

//         input             input
//         /   \               |
//      Norm   Mul    ==>     Mul (expect to be fused into the input layer)
//        |     |            /   \_
//      op_a   op_b       Norm   op_b
//                          |
//                        op_a
class ov::pass::activations_scaling::MulShareTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MulShareTransformation", "0");
    MulShareTransformation();
};

//        input_b   scalar        input_a   input_b
//              \   /                   \   /
//    input_a   Mul_b       ==>         Mul_a'  scalar
//          \   /                         \     /
//          Mul_a                          Mul_b' (expect to be merged with Mul_a')
class ov::pass::activations_scaling::MoveDownScalarMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveDownScalarMul", "0");
    MoveDownScalarMul();
};
