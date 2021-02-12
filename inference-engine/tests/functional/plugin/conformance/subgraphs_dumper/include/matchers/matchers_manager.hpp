// Copyright (C) 2021 Intel Corporation
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

    bool match_all(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref);

    bool match_any(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref);

    // TODO: Implement default xml config file generation by Matchers
    void generate_config() {};

private:
    std::vector<bool> run_matchers(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref);
// TODO: No copy constructor for xml_document
//    pugi::xml_document m_cfg;
    registry_map m_registry = {
            {SingleOpMatcher::name, []() { return std::make_shared<SingleOpMatcher>(); }}
    };
    matchers_map m_matchers = {};
};
}  // namespace SubgraphsDumper