// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matchers/base_matcher.hpp"
#include "cache/meta/input_info.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class MatchersManager {
public:
    using MatchersMap = std::map<std::string, BaseMatcher::Ptr>;

    explicit MatchersManager(const MatchersMap& matchers = {}) : m_matchers(matchers) {}

    bool match(const std::shared_ptr<ov::Node> &node,
               const std::shared_ptr<ov::Node> &ref);
    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref_model);

    std::list<ExtractedPattern> run_extractors(const std::shared_ptr<ov::Model> &model);

    void set_matchers(const MatchersMap& matchers = {}) { m_matchers = matchers; }
    const MatchersMap& get_matchers() { return m_matchers; }
    iMatcherConfig::Ptr get_config(const std::shared_ptr<ov::Node> &node) const;

private:
    MatchersMap m_matchers = {};
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
