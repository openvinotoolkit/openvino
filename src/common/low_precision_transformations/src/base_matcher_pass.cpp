// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/base_matcher_pass.hpp"
#include "openvino/core/node.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"

using namespace ov;
using namespace ov::pass::low_precision;

ov::pass::low_precision::BaseMatcherPass::BaseMatcherPass(const AttributeParameters& params) : params(params) {
}
