// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DisableShapeOfConstantFolding;

}  // namespace pass
}  // namespace ov

class ov::pass::DisableShapeOfConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableShapeOfConstantFolding");
    explicit DisableShapeOfConstantFolding(bool check_shape = true);
};
