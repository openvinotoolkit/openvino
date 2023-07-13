// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/op/util/op_types.hpp"

#include "common_test_utils/graph_comparator.hpp"

#include "matchers/base_matcher.hpp"
#include "matchers/manager.hpp"
#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class SubgraphMatcher : public BaseMatcher {
public:

    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref_model) const override;

protected:
    FunctionsComparator comparator = FunctionsComparator::no_default()
        .enable(FunctionsComparator::ATTRIBUTES)
        .enable(FunctionsComparator::NODES)
        .enable(FunctionsComparator::PRECISIONS);
    
    inline bool is_node_to_skip(const std::shared_ptr<ov::Node>& node) const {
        return ov::op::util::is_parameter(node) ||
               ov::op::util::is_constant(node) ||
               ov::op::util::is_output(node);
    }
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
