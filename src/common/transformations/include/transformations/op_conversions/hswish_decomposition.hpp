// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API HSwishDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishDecomposition transformation into sub-graph x * (min(Relu(x + 3), 6) * const(1/6).
 */
class ov::pass::HSwishDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("HSwishDecomposition", "0");
    HSwishDecomposition();
};
