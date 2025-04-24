// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertSubtract;
class TRANSFORMATIONS_API ConvertSubtractWithConstant;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertSubtract : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertSubtract");
    ConvertSubtract();
};

class ov::pass::ConvertSubtractWithConstant : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertSubtractWithConstant");
    ConvertSubtractWithConstant();
};
