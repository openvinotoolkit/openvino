// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/repeat_pattern.hpp"

using namespace ov::tools::subgraph_dumper;

std::list<ExtractedPattern>
RepeatPatternMatcher::extract(const std::shared_ptr<ov::Model> &model) {
    return {};
}

