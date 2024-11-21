// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SwiGLUFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SwiGLUFusion", "0");
    SwiGLUFusion();
};

}  // namespace pass
}  // namespace ov
