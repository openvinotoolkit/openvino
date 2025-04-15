// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matchers/subgraph/subgraph.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class ExtractorsManager {
public:
    using ExtractorsMap = std::map<std::string, SubgraphExtractor::Ptr>;

    explicit ExtractorsManager(const ExtractorsMap& extractors = {}) : m_extractors(extractors) {}

    std::vector<SubgraphExtractor::ExtractedPattern>
    extract(const std::shared_ptr<ov::Model> &model,
            bool is_extract_body = true,
            bool is_copy_constants = true);

    void set_extractors(const ExtractorsMap& extractors = {}) { m_extractors = extractors; }
    ExtractorsMap get_extractors() { return m_extractors; }

protected:
    ExtractorsMap m_extractors = {};
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
