// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class RemoveConvertBeforeGather : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveConvertBeforeGather", "0");
    RemoveConvertBeforeGather();
};

}   // namespace intel_gpu
}   // namespace ov
