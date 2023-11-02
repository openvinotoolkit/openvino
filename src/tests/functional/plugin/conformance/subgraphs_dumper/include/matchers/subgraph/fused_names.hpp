// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "matchers/subgraph/subgraph.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class FusedNamesExtractor final : public SubgraphExtractor {
public:
    FusedNamesExtractor(const std::string& device = "");

    std::vector<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &modele) override;

protected:
    std::unordered_set<std::string> extract_compiled_model_names(const std::shared_ptr<ov::Model>& model);
    void set_target_device(const std::string& _device);

    std::string device;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
