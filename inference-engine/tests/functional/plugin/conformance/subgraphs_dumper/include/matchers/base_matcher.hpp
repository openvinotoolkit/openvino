// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pugixml.hpp"
#include "ngraph/node.hpp"

namespace SubgraphsDumper {

class MatchersManager;

template<typename ...OPTypes>
struct MatcherConfig {
    MatcherConfig() : target_ops({}), ignored_ports({}), ignored_attributes({}) {}

    MatcherConfig(const std::vector<std::string> &_target_ops,
                  const std::vector<std::string> &_ignored_attributes,
                  const std::vector<size_t> &_ignored_ports) : target_ops(_target_ops),
                                                               ignored_attributes(_ignored_attributes),
                                                               ignored_ports(_ignored_ports) {}

    // Empty vectors stands for any of possible values
    std::vector<std::string> target_ops;  // extend to handle operation version
    std::vector<std::string> ignored_attributes;
    std::vector<size_t> ignored_ports;

    bool op_in_config(const std::shared_ptr<ngraph::Node> &node) {
        std::initializer_list<bool> vals{(ngraph::is_type<OPTypes>(node))...};
        return std::any_of(vals.begin(), vals.end(), [](bool i) { return i; });
    };
};

class Matcher {
    using Ptr = std::shared_ptr<Matcher>;
    friend MatchersManager;
public:
    virtual bool match(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) = 0;

    static const char *name;
protected:
    virtual const char *get_name() = 0;

    virtual void configure(const pugi::xml_document &cfg) = 0;

    void validate_and_unwrap_config();

    std::tuple<MatcherConfig<>> default_configs;
    std::map<std::string, MatcherConfig<>> matcher_configs_unwraped;
};
} // namespace SubgraphsDumper