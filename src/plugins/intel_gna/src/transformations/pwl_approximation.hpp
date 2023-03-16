// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
/**
 * @ingroup ie_transformation_common_api
 * @brief PWLApproximation transformation replaces suitable activation function with pwl.
 * It handles both cases with FakeQuntize as input and without it.
 */
class PWLApproximation : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PWLApproximation(const PWLApproximationMode& mode);
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
