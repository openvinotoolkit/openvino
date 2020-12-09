// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pugixml.hpp"
#include "ngraph/node.hpp"
#include "single_op.hpp"

namespace SubgraphsDumper {
class Matcher;

class MatchersManager {
public:
    typedef std::map<std::string, std::function<Matcher::Ptr()>> registry_map;
    typedef std::map<std::string, Matcher::Ptr> matchers_map;

    explicit MatchersManager(const std::string &cfg_path = "");

    bool match_all(std::shared_ptr<ngraph::Node> &node, std::shared_ptr<ngraph::Node> &ref);

    bool match_any(std::shared_ptr<ngraph::Node> &node, std::shared_ptr<ngraph::Node> &ref);

private:
    std::vector<bool> run_matchers(std::shared_ptr<ngraph::Node> &node, std::shared_ptr<ngraph::Node> &ref);

    pugi::xml_document m_cfg;
    registry_map m_registry = {
            {SingleOpMatcher::name, []() { return std::make_shared<SingleOpMatcher>(); }}
    };
    matchers_map m_matchers = {};
};
}  // namespace SubgraphsDumper