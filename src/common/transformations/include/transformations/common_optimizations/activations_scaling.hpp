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
class TRANSFORMATIONS_API ScaleDownSingleLayer;
class TRANSFORMATIONS_API MulGroupNormFusion;
class TRANSFORMATIONS_API MulMulAddFusion;

}  // namespace pass
}  // namespace ov

// ActivationsScaling scales down activations to prevent overflow due to the limited range of FP16
class ov::pass::ActivationsScaling : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ActivationsScaling", "0");
    explicit ActivationsScaling(float scale_factor) : m_scale_factor(scale_factor) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    float m_scale_factor = 0.f;
};

class ov::pass::ScaleDownSingleLayer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ScaleDownSingleLayer", "0");
    ScaleDownSingleLayer(float scale_factor);
};

class ov::pass::MulGroupNormFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulGroupNormFusion", "0");
    MulGroupNormFusion();
};

class ov::pass::MulMulAddFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MulMulAddFusion", "0");
    MulMulAddFusion();
};
