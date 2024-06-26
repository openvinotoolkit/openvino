// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class MatMulDecomposition: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MatMulDecomposition", "0");
    MatMulDecomposition();
};

}   // namespace intel_cpu
}   // namespace ov
