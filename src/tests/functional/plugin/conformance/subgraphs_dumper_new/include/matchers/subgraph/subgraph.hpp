// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "common_test_utils/graph_comparator.hpp"
#include "matchers/base_matcher.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class SubgraphMatcher : public BaseMatcher {
public:
    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref_model) const override;

protected:
    FunctionsComparator comparator = FunctionsComparator::with_default()
        .enable(FunctionsComparator::ATTRIBUTES)
        .enable(FunctionsComparator::NODES)
        .enable(FunctionsComparator::PRECISIONS)
        .enable(FunctionsComparator::ATTRIBUTES)
        .enable(FunctionsComparator::SUBGRAPH_DESCRIPTORS);;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
