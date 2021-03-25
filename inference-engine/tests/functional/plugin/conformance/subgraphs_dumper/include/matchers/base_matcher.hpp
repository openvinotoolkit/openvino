// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "ngraph/node.hpp"
#include "pugixml.hpp"

namespace SubgraphsDumper {

class MatchersManager;

class iMatcherConfig {
public:
    using Ptr = std::shared_ptr<iMatcherConfig>;

    explicit iMatcherConfig(bool is_fallback_config) : is_fallback_config(is_fallback_config) {}

    iMatcherConfig(
        std::vector<std::string> ignored_attributes,
        std::vector<size_t> ignored_ports,
        bool _is_fallback_config)
        : ignored_attributes(std::move(ignored_attributes)),
          ignored_ports(std::move(ignored_ports)),
          is_fallback_config(_is_fallback_config) {}

    // Empty vectors stands for any of possible values
    std::vector<std::string> ignored_attributes;
    std::vector<size_t> ignored_ports;
    bool is_fallback_config;

    virtual bool op_in_config(const std::shared_ptr<ngraph::Node>& node) = 0;

    virtual ~iMatcherConfig() = default;
};

template <typename... OPTypes>
struct MatcherConfig : public iMatcherConfig {
public:
    MatcherConfig() : iMatcherConfig(sizeof...(OPTypes) == 0) {}

    MatcherConfig(std::vector<std::string> ignored_attributes, std::vector<size_t> ignored_ports)
        : iMatcherConfig(
              std::move(ignored_attributes), std::move(ignored_ports), sizeof...(OPTypes) == 0) {}

    bool op_in_config(const std::shared_ptr<ngraph::Node> &node) override {
        std::initializer_list<bool> vals{(ngraph::is_type<OPTypes>(node))...};
        return std::any_of(vals.begin(), vals.end(), [](bool i) { return i; });
    };
};

class Matcher {
    using Ptr = std::shared_ptr<Matcher>;
    friend class MatchersManager;

public:
    virtual bool match(const std::shared_ptr<ngraph::Node>& node, const std::shared_ptr<ngraph::Node>& ref) const = 0;

    virtual ~Matcher() = default;

protected:
    virtual const char* get_name() = 0;

    virtual void configure(const pugi::xml_document& cfg) = 0;

    iMatcherConfig::Ptr get_config(const std::shared_ptr<ngraph::Node>& node) const;

    std::vector<iMatcherConfig::Ptr> default_configs;
};
}  // namespace SubgraphsDumper
