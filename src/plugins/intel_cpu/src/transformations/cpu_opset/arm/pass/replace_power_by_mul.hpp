// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ReplacePowerByMul: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReplacePowerByMul");
    ReplacePowerByMul();
};
}  // namespace intel_cpu
}  // namespace ov
