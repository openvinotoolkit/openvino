// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class ConvertGatherToGatherCompressed: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGatherToGatherCompressed", "0");
    ConvertGatherToGatherCompressed();
};

}   // namespace intel_gpu
}   // namespace ov
