// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class CausalMaskPreprocessFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("CausalMaskPreprocessFusion");
    CausalMaskPreprocessFusion();
};

}  // namespace intel_cpu
}  // namespace ov
