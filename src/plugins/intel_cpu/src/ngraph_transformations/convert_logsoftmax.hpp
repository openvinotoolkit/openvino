// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
class ConvertLogSoftmax: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertLogSoftmax", "0");
    ConvertLogSoftmax();
};

}   // namespace intel_cpu
}   // namespace ov
