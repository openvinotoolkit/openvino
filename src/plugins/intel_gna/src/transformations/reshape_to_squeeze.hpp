// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class ReshapeFuse : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeFuse", "0");
    ReshapeFuse();
};

class ReshapeToSqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeToSqueeze", "0");
    ReshapeToSqueeze();
};

class ReshapeToUnsqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeToUnsqueeze", "0");
    ReshapeToUnsqueeze();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
