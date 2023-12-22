// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"

namespace ov {
namespace intel_gpu {

class ConvertAvgPoolingToReduce : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertAvgPoolingToReduce", "0");
    ConvertAvgPoolingToReduce();
};

}  // namespace intel_gpu
}  // namespace ov
