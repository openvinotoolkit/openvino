// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Removes reshapes before MaxPool which do nothing. Such reshapes can be a result of conversion from IR10 to
 * IR7.
 */
class RemoveExtraReshapes : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveExtraReshapes", "0");
    RemoveExtraReshapes();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
