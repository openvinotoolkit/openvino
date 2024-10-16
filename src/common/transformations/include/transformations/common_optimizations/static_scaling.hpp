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

}  // namespace pass
}  // namespace ov

class ov::pass::StaticScaling : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StaticScaling", "0");
    StaticScaling(float scale_factor = 0.f);
};