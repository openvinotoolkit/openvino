// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "matchers/single_op.hpp"
namespace SubgraphsDumper {
class ConvolutionsMatcher : public SingleOpMatcher {
public:
    ConvolutionsMatcher();

    bool match_inputs(const std::shared_ptr<ngraph::Node> &node,
                      const std::shared_ptr<ngraph::Node> &ref,
                      const LayerTestsUtils::OPInfo &op_info) const override;
    bool match(const std::shared_ptr<ngraph::Node> &node,
               const std::shared_ptr<ngraph::Node> &ref,
               const LayerTestsUtils::OPInfo &op_info) const override;

protected:
    bool match_only_configured_ops() const override { return true; }
};
} // namespace SubgraphsDumper