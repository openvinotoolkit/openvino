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
class TRANSFORMATIONS_API EliminateMultiplyScalar;
class TRANSFORMATIONS_API MulConcatTransformation;
class TRANSFORMATIONS_API NormMulTransformation;
class TRANSFORMATIONS_API MulMulTransformation;

}  // namespace activations_scaling
}  // namespace pass
}  // namespace ov

// ActivationsScaling makes activation values smaller to prevent overflow due to the limited range of FP16
// This feature is controlled by ov::hint::activations_scale_factor.
// For example, when this property is set as 16, activations are divided by 16.
// If ov::hint::activations_scale_factor is less than or equal to zero, it is disabled.

class ov::pass::activations_scaling::ScaleDownSingleLayer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ScaleDownSingleLayer", "0");
    ScaleDownSingleLayer(float scale_factor, ov::element::Type scaled_prec);
};

class ov::pass::activations_scaling::EliminateMultiplyScalar : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateMultiplyScalar", "0");
    EliminateMultiplyScalar();
};

class ov::pass::activations_scaling::MulConcatTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MulConcatTransformation", "0");
    MulConcatTransformation();
};

class ov::pass::activations_scaling::NormMulTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormMulTransformation", "0");
    NormMulTransformation();
};

class ov::pass::activations_scaling::MulMulTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MulMulTransformation", "0");
    MulMulTransformation();
};
