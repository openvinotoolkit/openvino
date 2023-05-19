// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface RemoveConverts
 * @brief Remove sequence of two ConvertSaturation operations for specific precisions: FP32 => BF16 => FP32
 * @ingroup snippets
 */
class RemoveConverts : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveConverts", "0");
    RemoveConverts();
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
