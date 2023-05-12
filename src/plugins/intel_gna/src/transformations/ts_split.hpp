// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief TODO
 */
class TSSplitBackward : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TSSplitBackward", "0");
    TSSplitBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
