// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pattern/op/block.hpp"

namespace ov::intel_cpu {

class ConvMulAddFQBlock : public ov::pass::pattern::op::Block {
public:
    explicit ConvMulAddFQBlock(bool require_int_fq_output);
};

}  // namespace ov::intel_cpu
