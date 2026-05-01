// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "../v1/subgraph_pipeline.hpp"

namespace ov {
namespace npuw {
namespace attn {

std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> register_patterns(
    ov::npuw::v1::subgraphs::PatternRegistry& registry);

}  // namespace attn
}  // namespace npuw
}  // namespace ov
