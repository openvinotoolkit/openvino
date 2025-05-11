// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matchers/single_op/single_op.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class MatchersManager {
public:
    using MatchersMap = std::map<std::string, SingleOpMatcher::Ptr>;
    explicit MatchersManager(const MatchersMap& matchers = {}) : m_matchers(matchers) {}

    bool match(const std::shared_ptr<ov::Node> &node,
               const std::shared_ptr<ov::Node> &ref) const;

    void set_matchers(const MatchersMap& matchers = {}) { m_matchers = matchers; }
    void set_shape_strict_match(bool shape_strict_match);
    void set_match_attributes(bool match_attribute);
    void set_match_in_types(bool match_in_types);

    const MatchersMap& get_matchers() { return m_matchers; }
    iMatcherConfig::Ptr get_config(const std::shared_ptr<ov::Node> &node) const;

protected:
    MatchersMap m_matchers = {};
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
