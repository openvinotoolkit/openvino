// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "../v1/subgraph_pipeline.hpp"

namespace ov {
namespace npuw {
namespace attn {

enum class BehaviorKind { Dynamic, Pyramid, HFA };

std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> register_patterns(
    ov::npuw::v1::subgraphs::PatternRegistry& registry);

void attach_runtime_behavior(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                             ov::npuw::v1::subgraphs::Context& compiled_context,
                             BehaviorKind kind);

}  // namespace attn
}  // namespace npuw
}  // namespace ov
