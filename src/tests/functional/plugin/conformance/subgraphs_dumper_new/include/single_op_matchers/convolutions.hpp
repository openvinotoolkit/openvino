// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op_matchers/base.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class ConvolutionsMatcher : public virtual BaseMatcher {
public:
    ConvolutionsMatcher();

    bool match_inputs(const std::shared_ptr<ov::Node> &node,
                      const std::shared_ptr<ov::Node> &ref) const override;
    bool match(const std::shared_ptr<ov::Node> &node,
               const std::shared_ptr<ov::Node> &ref) const override;

protected:
    bool match_only_configured_ops() const override { return true; }
};
}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
