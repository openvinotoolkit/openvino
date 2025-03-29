// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matchers/subgraph/subgraph.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class ReadValueAssignExtractor final : public SubgraphExtractor {
public:
    std::vector<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) override;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
