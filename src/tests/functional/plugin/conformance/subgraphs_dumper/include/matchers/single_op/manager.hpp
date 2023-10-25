// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matchers/single_op/single_op.hpp"
#include "cache/meta/input_info.hpp"

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

    const MatchersMap& get_matchers() { return m_matchers; }
    iMatcherConfig::Ptr get_config(const std::shared_ptr<ov::Node> &node) const;

protected:
    MatchersMap m_matchers = {};
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
