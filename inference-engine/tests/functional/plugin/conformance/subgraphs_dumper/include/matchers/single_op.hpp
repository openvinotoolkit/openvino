// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "base_matcher.hpp"
#include "pugixml.hpp"
#include "ngraph/node.hpp"

namespace SubgraphsDumper {

class SingleOpMatcher : public Matcher {
public:
    bool match(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) override;
    static const char *name;
private:
    void configure(const pugi::xml_document &cfg) override {};
    std::vector<size_t> ignore_const_port_vals = {0};
};
}  // namespace SubgraphsDumper