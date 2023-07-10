// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "matchers/subgraph/subgraph.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class RepeatPatternMatcher : public SubgraphMatcher {
public:
    std::list<BaseMatcher::ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) override;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
