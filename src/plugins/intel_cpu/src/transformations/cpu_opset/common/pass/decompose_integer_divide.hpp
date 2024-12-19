// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_cpu {

class DecomposeIntegerDivide : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeIntegerDivide", "0");
    DecomposeIntegerDivide();
};

}  // namespace intel_cpu
}  // namespace ov
