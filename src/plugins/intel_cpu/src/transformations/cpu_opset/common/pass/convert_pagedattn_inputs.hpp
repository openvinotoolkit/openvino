// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "config.h"

namespace ov {
namespace intel_cpu {

class ConvertPagedAttnInputs : public ov::pass::MatcherPass {
public:    
    OPENVINO_MATCHER_PASS_RTTI("ConvertPagedAttnInputs");
    ConvertPagedAttnInputs(const Config& p);
private:
    const Config& config;
};

}  // namespace intel_cpu
}  // namespace ov
