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
class TRANSFORMATIONS_API ScaleDownMultipleLayers;
class TRANSFORMATIONS_API MulGroupNormTransformation;
class TRANSFORMATIONS_API MulMulAddTransformation;
class TRANSFORMATIONS_API SplitTransformation;
class TRANSFORMATIONS_API ReshapeTransformation;
class TRANSFORMATIONS_API MulMulMulTransformation;
class TRANSFORMATIONS_API MulMVNTransformation;
class TRANSFORMATIONS_API ConcatTransformation;

}  // namespace activations_scaling
}  // namespace pass
}  // namespace ov

// ActivationsScaling makes activation values smaller to prevent overflow due to the limited range of FP16
// This feature is controlled by ov::hint::activations_scale_factor.
// For example, when this property is set as 16, activations are divided by 16.
// If ov::hint::activations_scale_factor is less than zero, it is disabled.
class ov::pass::ActivationsScaling : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ActivationsScaling", "0");
    explicit ActivationsScaling(float scale_factor, ov::element::Type scaled_prec)
    : m_scale_factor(scale_factor), m_scaled_prec(scaled_prec) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    float m_scale_factor = 0.f;
    ov::element::Type m_scaled_prec = element::f16;
};

class ov::pass::activations_scaling::ScaleDownSingleLayer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ScaleDownSingleLayer", "0");
    ScaleDownSingleLayer(float scale_factor, ov::element::Type scaled_prec);
};

class ov::pass::activations_scaling::ScaleDownMultipleLayers : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ScaleDownMultipleLayers", "0");
    ScaleDownMultipleLayers(float scale_factor);
};

class ov::pass::activations_scaling::MulGroupNormTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulGroupNormTransformation", "0");
    MulGroupNormTransformation();
};

class ov::pass::activations_scaling::MulMulAddTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulMulAddTransformation", "0");
    MulMulAddTransformation();
};

class ov::pass::activations_scaling::SplitTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitTransformation", "0");
    SplitTransformation();
};

class ov::pass::activations_scaling::ReshapeTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeTransformation", "0");
    ReshapeTransformation();
};

class ov::pass::activations_scaling::MulMulMulTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulMulMulTransformation", "0");
    MulMulMulTransformation();
};

class ov::pass::activations_scaling::MulMVNTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulMVNTransformation", "0");
    MulMVNTransformation();
};

class ov::pass::activations_scaling::ConcatTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConcatTransformation", "0");
    ConcatTransformation();
};
