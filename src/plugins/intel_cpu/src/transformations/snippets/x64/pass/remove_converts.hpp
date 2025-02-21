// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface RemoveConverts
 * @brief Remove sequence of two ConvertSaturation operations for specific precisions: FP32 => BF16 => FP32
 * @ingroup snippets
 */
class RemoveConverts : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RemoveConverts");
    RemoveConverts();
};

}  // namespace ov::intel_cpu::pass
