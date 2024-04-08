// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RNNCellTfKerasFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief RNNCellFusion transformation incorporates
 * the sequence of operation within the RNNCell operation.
 */
class ov::pass::RNNCellTfKerasFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RNNCellTfKerasFusion", "0");
    RNNCellTfKerasFusion();
};