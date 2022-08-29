// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Decompose Concat operation
 * Some types of Concat operations which are not supported
 * natively by GNA HW are handled by this decomposition.
 * 
 */
class DecomposeConcat : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeConcat", "0");
    DecomposeConcat();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
