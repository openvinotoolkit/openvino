// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class ConvertBinaryConvolutionToConvolution: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertBinaryConvolutionToConvolution");
    ConvertBinaryConvolutionToConvolution();
};

}   // namespace ov::intel_gpu
