// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pugixml.hpp"
#include "ngraph/node.hpp"

namespace SubgraphsDumper {

class MatchersManager;

class iMatcherConfig {
public:
    using Ptr = std::shared_ptr<iMatcherConfig>;

    // Empty vectors stands for any of possible values
    std::vector<std::string> ignored_attributes;
    std::vector<size_t> ignored_ports;
    bool is_fallback_config;

    virtual bool op_in_config(const std::shared_ptr<ngraph::Node> &node) = 0;
};

template<typename ...OPTypes>
struct MatcherConfig : public iMatcherConfig {
public:
    MatcherConfig() {
        if (sizeof...(OPTypes) == 0) {
            is_fallback_config = true;
        }
    }
    MatcherConfig(const std::vector<std::string> &_ignored_attributes,
                  const std::vector<size_t> &_ignored_ports) : ignored_attributes(_ignored_attributes),
                                                               ignored_ports(_ignored_ports) {
        if (sizeof...(OPTypes) == 0) {
            is_fallback_config = true;
        }
    }

    // Empty vectors stands for any of possible values
    std::vector<std::string> ignored_attributes = {};
    std::vector<size_t> ignored_ports = {};
    bool is_fallback_config = false;

    bool op_in_config(const std::shared_ptr<ngraph::Node> &node) override {
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

    iMatcherConfig::Ptr get_config(const std::shared_ptr<ngraph::Node> &node);

    std::vector<iMatcherConfig::Ptr> default_configs;
};
} // namespace SubgraphsDumper