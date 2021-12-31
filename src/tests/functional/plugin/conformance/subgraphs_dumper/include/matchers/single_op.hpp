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

    bool match(const std::shared_ptr<ngraph::Node> &node,
               const std::shared_ptr<ngraph::Node> &ref,
               const LayerTestsUtils::OPInfo &op_info) const override;
    bool same_op_type(const std::shared_ptr<ngraph::Node> &node,
                      const std::shared_ptr<ngraph::Node> &ref,
                      const LayerTestsUtils::OPInfo &op_info) const;

    virtual bool match_inputs(const std::shared_ptr<ngraph::Node> &node,
                      const std::shared_ptr<ngraph::Node> &ref,
                      const LayerTestsUtils::OPInfo &op_info) const;
    bool match_outputs(const std::shared_ptr<ngraph::Node> &node,
                       const std::shared_ptr<ngraph::Node> &ref,
                       const LayerTestsUtils::OPInfo &op_info) const;
    bool same_attrs(const std::shared_ptr<ngraph::Node> &node,
                    const std::shared_ptr<ngraph::Node> &ref,
                    const LayerTestsUtils::OPInfo &op_info) const;
    bool match_ports(const std::shared_ptr<ngraph::Node> &node,
                     const std::shared_ptr<ngraph::Node> &ref,
                     const LayerTestsUtils::OPInfo &op_info) const;

protected:
    void configure(const pugi::xml_document &cfg) override {}

    bool match_only_configured_ops() const override { return false; }
};
}  // namespace SubgraphsDumper
