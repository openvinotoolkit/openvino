// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace intel_cpu {

class FullyConnectedBiasFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FullyConnectedBiasFusion");
    FullyConnectedBiasFusion();
};

}  // namespace intel_cpu
}  // namespace ov
