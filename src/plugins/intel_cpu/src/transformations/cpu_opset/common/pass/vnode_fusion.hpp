// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class VNodeFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("VNodeFusion", "0");
    VNodeFusion();
};

}  // namespace intel_cpu
}  // namespace ov