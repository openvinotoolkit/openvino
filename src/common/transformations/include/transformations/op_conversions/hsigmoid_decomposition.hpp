// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API HSigmoidDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief HSigmoidDecomposition transformation into sub-graph (min(Relu(x + 3), 6) * const(1/6).
 */
class ov::pass::HSigmoidDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSigmoidDecomposition", "0");
    HSigmoidDecomposition();
};
