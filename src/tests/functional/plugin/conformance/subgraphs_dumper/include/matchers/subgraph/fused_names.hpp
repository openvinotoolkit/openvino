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
    ~FusedNamesExtractor();

    std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model,
                                        bool is_extract_body = true,
                                        bool is_copy_constants = true) override;

protected:
    std::unordered_set<std::string> extract_compiled_model_names(const std::shared_ptr<ov::Model>& model);
    void set_target_device(const std::string& _device);

    std::string device;
    std::shared_ptr<ov::Core> core;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
