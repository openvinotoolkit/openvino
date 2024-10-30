// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class DecomposeRMSNorm: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeRMSNorm", "0");
    DecomposeRMSNorm();
};

}   // namespace intel_cpu
}   // namespace ov
