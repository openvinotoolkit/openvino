// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class MVN6PowerDecomposition: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MVN6PowerDecomposition");
    MVN6PowerDecomposition();
};
}  // namespace intel_cpu
}  // namespace ov
