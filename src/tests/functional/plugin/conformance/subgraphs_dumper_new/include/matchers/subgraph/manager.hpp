// Copyright (C) 2018-2023 Intel Corporation
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

    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref);
    std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model,
                                        bool is_extract_body = true);

    void set_extractors(const ExtractorsMap& extractors = {}) { m_extractors = extractors; }
    ExtractorsMap get_extractors() { return m_extractors; }

protected:
    ExtractorsMap m_extractors = {};
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
