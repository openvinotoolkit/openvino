// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "pugixml.hpp"

#include "matchers/config.hpp"
#include "cache/meta/input_info.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class BaseMatcher {
public:
    using Ptr = std::shared_ptr<BaseMatcher>;

    BaseMatcher() = default;

    virtual bool match(const std::shared_ptr<ov::Node> &node,
                       const std::shared_ptr<ov::Node> &ref) const {};
    virtual bool match(const std::shared_ptr<ov::Model> &model,
                       const std::shared_ptr<ov::Model> &ref_model) const {};

    virtual std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) {};

    iMatcherConfig::Ptr get_config(const std::shared_ptr<ov::Node> &node) const;

protected:
    std::vector<iMatcherConfig::Ptr> default_configs;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
