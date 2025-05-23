// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class ConvertDequantizationFP16: public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertDequantizationFP16");
    ConvertDequantizationFP16(const element::TypeVector& precisions);
};

/**
 * @brief 

 */
class ConvertDequantizationFP16Matcher: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertDequantizationFP16Matcher");
    ConvertDequantizationFP16Matcher(const element::TypeVector& precisions);
};

}   // namespace ov::intel_gpu
