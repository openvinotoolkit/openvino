// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API StaticScaling;
class TRANSFORMATIONS_API StaticScalingInput;
class TRANSFORMATIONS_API StaticScalingOutput;
class TRANSFORMATIONS_API StaticScalingAdd;
class TRANSFORMATIONS_API StaticScalingModel;

}  // namespace pass
}  // namespace ov

class ov::pass::StaticScalingModel : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("StaticScalingModel", "0");
    explicit StaticScalingModel(float scale_factor): m_scale_factor(scale_factor) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    float m_scale_factor = 0.f;
};
