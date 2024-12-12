// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace intel_cpu {

class ConvertMatMulToFC : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMatMulToFC");
    ConvertMatMulToFC();
};

}  // namespace intel_cpu
}  // namespace ov
