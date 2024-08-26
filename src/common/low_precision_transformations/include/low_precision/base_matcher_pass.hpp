// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/node.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "rt_info/attribute_parameters.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API BaseMatcherPass;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

class LP_TRANSFORMATIONS_API ov::pass::low_precision::BaseMatcherPass : public ov::pass::MatcherPass {
public:
    BaseMatcherPass(const AttributeParameters& params = AttributeParameters());
    AttributeParameters params;
};
