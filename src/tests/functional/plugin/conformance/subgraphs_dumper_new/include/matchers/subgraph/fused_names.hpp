// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "matchers/subgraph/subgraph.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class FusedNamesMatcher : public SubgraphMatcher {
public:
    std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) override;

private:
    std::unordered_set<std::string> extract_compiled_model_names(const std::shared_ptr<ov::Model>& model);
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
