// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace pass {

class PrimPlaceholderReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::jax::pass::Placeholder");
    PrimPlaceholderReplacer();
};

// This pass is just a fake one, needs to be removed in the future.
class PlaceholderInBodyReplacer : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::jax::pass::Placeholder");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace pass
}  // namespace jax
}  // namespace frontend
}  // namespace ov