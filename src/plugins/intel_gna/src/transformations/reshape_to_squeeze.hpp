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

class ReshapeReduction : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ReshapeReduction", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
