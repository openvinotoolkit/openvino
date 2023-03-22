// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GatherSinkingReshapeBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingReshapeBackward", "0");
    GatherSinkingReshapeBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
