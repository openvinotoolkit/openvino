// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class BigPatternFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("BigPatternFusion", "0");
    BigPatternFusion();
};

}  // namespace intel_cpu
}  // namespace ov