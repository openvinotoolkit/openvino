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

TRANSFORMATIONS_API void mark_as_scale_down_node(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_scale_down_node(const std::shared_ptr<const Node>& node);

class TRANSFORMATIONS_API ScaleDownNode : public RuntimeAttribute {
public:
    OPENVINO_RTTI("scale_down_node", "0");

    bool is_copyable() const override {
        return false;
    }
};

class TRANSFORMATIONS_API ScaleDownSingleLayer;
class TRANSFORMATIONS_API ScaleDownFusion;
class TRANSFORMATIONS_API EliminateMultiplyNorm;
class TRANSFORMATIONS_API MulConcatTransformation;
class TRANSFORMATIONS_API NormMulTransformation;
class TRANSFORMATIONS_API EliminateMultiplyX1;

}  // namespace activations_scaling
}  // namespace pass
}  // namespace ov

// ActivationsScaling makes activation values smaller to prevent overflow due to the limited range of FP16
// This feature is controlled by ov::hint::activations_scale_factor.
// For example, when this property is set as 16, activations are divided by 16.
// If ov::hint::activations_scale_factor is less than zero, it is disabled.

class ov::pass::activations_scaling::ScaleDownSingleLayer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ScaleDownSingleLayer", "0");
    ScaleDownSingleLayer(float scale_factor, ov::element::Type scaled_prec);
};

class ov::pass::activations_scaling::ScaleDownFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ScaleDownFusion", "0");
    ScaleDownFusion();
};

class ov::pass::activations_scaling::EliminateMultiplyNorm : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateMultiplyNorm", "0");
    EliminateMultiplyNorm();
};

class ov::pass::activations_scaling::MulConcatTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulConcatTransformation", "0");
    MulConcatTransformation();
};

class ov::pass::activations_scaling::NormMulTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NormMulTransformation", "0");
    NormMulTransformation();
};

class ov::pass::activations_scaling::EliminateMultiplyX1 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateMultiplyX1", "0");
    EliminateMultiplyX1();
};
