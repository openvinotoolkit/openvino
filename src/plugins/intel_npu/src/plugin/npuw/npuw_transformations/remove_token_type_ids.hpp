// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Drops the `token_type_ids` parameter and the subgraphs it feeds from a generate model.
// The parameter is only removed once every one of its consumers has been detached, so there are
// three possible outcomes:
//   * no known subgraph is found - the model is left untouched and the pass returns false;
//   * every consumer is detached  - the parameter is removed and the pass returns true;
//   * some consumer survives the rewrites - the model is now partially transformed and cannot be
//     left that way, so the pass throws.
class RemoveTokenTypeIds : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::RemoveTokenTypeIds");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace ov::npuw
