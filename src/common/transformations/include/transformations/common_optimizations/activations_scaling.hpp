// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
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
