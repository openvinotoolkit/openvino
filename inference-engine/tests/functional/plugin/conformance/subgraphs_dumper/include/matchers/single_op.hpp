// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "base_matcher.hpp"
#include "pugixml.hpp"
#include "ngraph/node.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

namespace SubgraphsDumper {


class SingleOpMatcher : public Matcher {
public:
    SingleOpMatcher();

    bool match(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) const override;

    static const char *name;
protected:
    const char *get_name() override { return name; }

    void configure(const pugi::xml_document &cfg) override {}
private:
    bool same_op_type(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) const;
    bool match_inputs(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) const;
    bool match_outputs(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) const;
    bool same_attrs(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) const;
    bool match_ports(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) const;
};
}  // namespace SubgraphsDumper
