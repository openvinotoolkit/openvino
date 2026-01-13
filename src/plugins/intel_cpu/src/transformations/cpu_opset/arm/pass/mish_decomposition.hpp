// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

class MishDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MishDecomposition");
    MishDecomposition();
};

}  // namespace ov::intel_cpu
