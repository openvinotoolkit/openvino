// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {
/**
 * @brief Pooling can be reordered with activation, on GNA there is a strategy to have conv->maxpool->activation
 * it means maxpool receives 4 bytes, and produces 4 bytes
 */
class ReorderActivationAndPooling : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReorderActivationAndPooling", "0");
    ReorderActivationAndPooling();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
