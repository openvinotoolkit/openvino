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

}  // namespace pass
}  // namespace ov

class ov::pass::ActivationsScaling : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ActivationsScaling", "0");
    explicit ActivationsScaling(float scale_factor): m_scale_factor(scale_factor) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    float m_scale_factor = 0.f;
};
