// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class FuseTruncDivide: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseTruncDivide", "0");
    FuseTruncDivide();
};

}   // namespace intel_cpu
}   // namespace ov
