// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/subgraph.hpp"

using namespace ov::tools::subgraph_dumper;

bool
SubgraphMatcher::match(const std::shared_ptr<ov::Model> &model,
                       const std::shared_ptr<ov::Model> &ref_model) const {
    return comparator.compare(model, ref_model).valid;
}
