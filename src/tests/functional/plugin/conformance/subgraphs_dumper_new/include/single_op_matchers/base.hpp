// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "pugixml.hpp"

#include "single_op_matchers/config.hpp"
#include "cache/meta.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class BaseMatcher {
public:
    using Ptr = std::shared_ptr<BaseMatcher>;
    BaseMatcher();

    virtual bool match(const std::shared_ptr<ov::Node> &node,
                       const std::shared_ptr<ov::Node> &ref) const;

protected:
    virtual void configure(const pugi::xml_document &cfg) {};
    // todo: iefode: remove?
    virtual bool match_only_configured_ops() const { return false; }; // TODO: Add setter for external configuration purposes.
    virtual bool match_inputs(const std::shared_ptr<ov::Node> &node,
                              const std::shared_ptr<ov::Node> &ref) const;
    virtual bool same_op_type(const std::shared_ptr<ov::Node> &node,
                              const std::shared_ptr<ov::Node> &ref) const;
    virtual bool match_outputs(const std::shared_ptr<ov::Node> &node,
                               const std::shared_ptr<ov::Node> &ref) const;
    virtual bool match_attrs(const std::shared_ptr<ov::Node> &node,
                             const std::shared_ptr<ov::Node> &ref) const;

    iMatcherConfig::Ptr get_config(const std::shared_ptr<ov::Node> &node) const;
    std::vector<iMatcherConfig::Ptr> default_configs;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
