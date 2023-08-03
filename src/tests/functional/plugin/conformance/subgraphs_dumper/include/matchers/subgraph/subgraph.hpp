// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/op/util/op_types.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "cache/meta/input_info.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class SubgraphExtractor {
public:
    using Ptr = std::shared_ptr<SubgraphExtractor>;

    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref_model) const;

    virtual std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model,
                                                bool is_extract_body = true) {
        return std::list<ExtractedPattern>{};
    };

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
