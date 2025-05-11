// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/core/node.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class iMatcherConfig {
public:
    using Ptr = std::shared_ptr<iMatcherConfig>;

    explicit iMatcherConfig(bool is_fallback_config) : is_fallback_config(is_fallback_config) {}

    iMatcherConfig(
            const std::vector<std::string>& ignored_attributes,
            const std::vector<size_t>& ignored_ports,
            bool is_fallback_config,
            bool ignore_matching = false) :
            ignored_attributes(ignored_attributes),
            ignored_ports(ignored_ports),
            is_fallback_config(is_fallback_config),
            ignore_matching(ignore_matching) {}

    // Empty vectors stands for any of possible values
    std::vector<std::string> ignored_attributes;
    std::vector<size_t> ignored_ports;
    bool is_fallback_config;
    bool ignore_matching = false;

    virtual bool op_in_config(const std::shared_ptr<ov::Node> &node) = 0;
};

template <typename... OPTypes>
struct MatcherConfig : public iMatcherConfig {
public:
    MatcherConfig() : iMatcherConfig(sizeof...(OPTypes) == 0) {}

    MatcherConfig(const std::vector<std::string>& ignored_attributes,
                  const std::vector<size_t>& ignored_ports,
                  bool ignore_matching = false) :
                  iMatcherConfig(ignored_attributes, ignored_ports, sizeof...(OPTypes) == 0, ignore_matching) {}

    MatcherConfig(bool ignore_matching) : iMatcherConfig({}, {}, sizeof...(OPTypes) == 0, ignore_matching) {}

    bool op_in_config(const std::shared_ptr<ov::Node> &node) override {
        std::initializer_list<bool> vals{(ov::is_type<OPTypes>(node))...};
        return std::any_of(vals.begin(), vals.end(), [](bool i) { return i; });
    };
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
