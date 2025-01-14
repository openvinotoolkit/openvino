// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <utility>

#include "pugixml.hpp"
#include "matchers/single_op/config.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class SingleOpMatcher {
public:
    using Ptr = std::shared_ptr<SingleOpMatcher>;
    SingleOpMatcher();

    virtual bool match(const std::shared_ptr<ov::Node> &node,
                       const std::shared_ptr<ov::Node> &ref) const;

    iMatcherConfig::Ptr get_config(const std::shared_ptr<ov::Node> &node) const;
    void set_strict_shape_match(bool strict_shape_match);
    void set_match_attrib(bool match_attrib);
    void set_match_in_types(bool match_in_types);

protected:
    virtual void configure(const pugi::xml_document &cfg) {};
    virtual bool match_only_configured_ops() const { return false; };
    virtual bool match_inputs(const std::shared_ptr<ov::Node> &node,
                              const std::shared_ptr<ov::Node> &ref) const;
    virtual bool same_op_type(const std::shared_ptr<ov::Node> &node,
                              const std::shared_ptr<ov::Node> &ref) const;
    virtual bool match_outputs(const std::shared_ptr<ov::Node> &node,
                               const std::shared_ptr<ov::Node> &ref) const;
    virtual bool match_attrs(const std::shared_ptr<ov::Node> &node,
                             const std::shared_ptr<ov::Node> &ref) const;

    std::vector<iMatcherConfig::Ptr> default_configs;
    // match only shape ranks by default;
    bool is_strict_shape_match = false;
    bool is_match_attributes = true;
    bool is_match_in_types = false;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
