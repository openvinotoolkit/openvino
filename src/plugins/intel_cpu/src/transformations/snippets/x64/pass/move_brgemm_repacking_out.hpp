// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

class MoveBrgemmRepackingOut: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MoveBrgemmRepackingOut", "0");
    MoveBrgemmRepackingOut();
};


}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
