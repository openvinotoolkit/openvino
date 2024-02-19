// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
class StatefulSDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StatefulSDPAFusion", "0");
    StatefulSDPAFusion();
};

}   // namespace intel_cpu
}   // namespace ov
