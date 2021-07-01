// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "ngraph/node.hpp"
#include "pugixml.hpp"

#include "functional_test_utils/include/functional_test_utils/layer_test_utils/op_info.hpp"

namespace SubgraphsDumper {

class MatchersManager;

class iMatcherConfig {
public:
    using Ptr = std::shared_ptr<iMatcherConfig>;

    explicit iMatcherConfig(bool is_fallback_config) : is_fallback_config(is_fallback_config) {}

    iMatcherConfig(
            std::vector<std::string> ignored_attributes,
            std::vector<size_t> ignored_ports,
            bool is_fallback_config,
            bool ignore_matching = false)
            : ignored_attributes(std::move(ignored_attributes)),
              ignored_ports(std::move(ignored_ports)),
              is_fallback_config(is_fallback_config),
              ignore_matching(ignore_matching) {}

    // Empty vectors stands for any of possible values
    std::vector<std::string> ignored_attributes;
    std::vector<size_t> ignored_ports;
    bool is_fallback_config;
    bool ignore_matching = false;

    virtual bool op_in_config(const std::shared_ptr<ngraph::Node> &node) = 0;

    virtual ~iMatcherConfig() = default;
};

template <typename... OPTypes>
struct MatcherConfig : public iMatcherConfig {
public:
    MatcherConfig() : iMatcherConfig(sizeof...(OPTypes) == 0) {}

    MatcherConfig(std::vector<std::string> ignored_attributes, std::vector<size_t> ignored_ports,
                  bool ignore_matching = false)
            : iMatcherConfig(
            std::move(ignored_attributes), std::move(ignored_ports), sizeof...(OPTypes) == 0, ignore_matching) {}

    MatcherConfig(bool ignore_matching) : iMatcherConfig({}, {}, sizeof...(OPTypes) == 0, ignore_matching) {}

    bool op_in_config(const std::shared_ptr<ngraph::Node> &node) override {
        std::initializer_list<bool> vals{(ngraph::is_type<OPTypes>(node))...};
        return std::any_of(vals.begin(), vals.end(), [](bool i) { return i; });
    };
};

class Matcher {
    using Ptr = std::shared_ptr<Matcher>;

    friend class MatchersManager;

public:
    virtual bool match(const std::shared_ptr<ngraph::Node> &node,
                       const std::shared_ptr<ngraph::Node> &ref,
                       const LayerTestsUtils::OPInfo &op_info) const = 0;

    virtual ~Matcher() = default;

protected:
    virtual void configure(const pugi::xml_document &cfg) = 0;

    iMatcherConfig::Ptr get_config(const std::shared_ptr<ngraph::Node> &node) const;

    std::vector<iMatcherConfig::Ptr> default_configs;

    virtual bool match_only_configured_ops() const = 0; // TODO: Add setter for external configuration purposes.
};
}  // namespace SubgraphsDumper
