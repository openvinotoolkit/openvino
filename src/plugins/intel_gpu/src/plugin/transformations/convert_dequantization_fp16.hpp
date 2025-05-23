// Copyright (C) 2025 Intel Corporation
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
 * @brief This transformation adds Convert primitive to dequantization subgraphs' endpoints
 * which is targeting to keep dequantization subgraphs' outputs aligned with GPU plugin's inference precision (mostly fp16)
 * to avoid performance regression from keeping dequantization subgraphs compute in fp32 in the KeepDequantizationPrecision pass. 
 * We assume that the values of the dequantization output will fit into the inference precision of the GPU plugin.
 */
class ConvertDequantizationFP16Matcher: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertDequantizationFP16Matcher");
    ConvertDequantizationFP16Matcher(const element::TypeVector& precisions);
};

}   // namespace ov::intel_gpu
