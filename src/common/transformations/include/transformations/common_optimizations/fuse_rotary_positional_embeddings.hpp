// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API RPE_Fusion;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Fuses special sub-graph into an internal Rotary Positional Embedding operation
 */
class ov::pass::RPE_Fusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RPE_Fusion", "0");
    RPE_Fusion();
};
