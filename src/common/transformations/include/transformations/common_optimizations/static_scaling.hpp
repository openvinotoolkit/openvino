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

class ov::pass::StaticScaling : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StaticScaling", "0");
    StaticScaling(float scale_factor = 0.f);
};

class ov::pass::StaticScalingInput : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StaticScalingInput", "0");
    StaticScalingInput(float scale_factor = 0.f);
};

class ov::pass::StaticScalingOutput : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StaticScalingOutput", "0");
    StaticScalingOutput(float scale_factor = 0.f);
};

class ov::pass::StaticScalingAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StaticScalingAdd", "0");
    StaticScalingAdd(float scale_factor = 0.f);
};

class ov::pass::StaticScalingModel : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("StaticScalingModel", "0");
    explicit StaticScalingModel(float scale_factor = 0.f);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    float m_scale_factor = 0.f;
};
