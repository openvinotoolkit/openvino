// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LSTMCellFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief LSTMCellFusion transformation replaces a sequence of
 * operations with LSTMCell op.
 */
class ov::pass::LSTMCellFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMCellFusion", "0");
    LSTMCellFusion();
};
