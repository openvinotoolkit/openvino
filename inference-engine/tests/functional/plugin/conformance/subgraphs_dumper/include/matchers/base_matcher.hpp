// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pugixml.hpp"
#include "ngraph/node.hpp"

namespace SubgraphsDumper {

class MatchersManager;

class Matcher {
    using Ptr = std::shared_ptr<Matcher>;
    friend MatchersManager;
public:
    virtual bool match(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) = 0;

    static const char *name;
protected:
    virtual void configure(const pugi::xml_document &cfg) = 0;
};
} // namespace SubgraphsDumper