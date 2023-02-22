// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GatherSinkingTransposeReshapeForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingTransposeReshapeForward", "0");
    GatherSinkingTransposeReshapeForward();
};

class GatherSinkingTransposeReshapeBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingTransposeReshapeBackward", "0");
    GatherSinkingTransposeReshapeBackward();
};

} // namespace pass
} // namespace intel_gna
} // namespace ov
