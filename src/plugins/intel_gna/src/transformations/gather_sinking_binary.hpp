// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GatherSinkingBinaryForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingBinaryForward", "0");
    GatherSinkingBinaryForward();
};

class GatherSinkingBinaryBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingBinaryBackward", "0");
    GatherSinkingBinaryBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov