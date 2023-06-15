// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include "pugixml.hpp"
// #include "ngraph/node.hpp"
// #include "single_op_matchers/single_op.hpp"
// #include "single_op_matchers/convolutions.hpp"

#include "single_op_matchers/base.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class Matcher;

class MatchersManager {
public:
    using MatchersMap = std::map<std::string, BaseMatcher::Ptr>;

    explicit MatchersManager(const MatchersMap& matchers = {}) : m_matchers(matchers) {}

    bool match_all(const std::shared_ptr<ov::Node> &node,
                   const std::shared_ptr<ov::Node> &ref);
    bool match_any(const std::shared_ptr<ov::Node> &node,
                   const std::shared_ptr<ov::Node> &ref);
    void set_matchers(const MatchersMap& matchers = {}) { m_matchers = matchers; }

private:
    std::vector<bool> run_matchers(const std::shared_ptr<ov::Node> &node,
                                   const std::shared_ptr<ov::Node> &ref);

    MatchersMap m_matchers = {};
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
