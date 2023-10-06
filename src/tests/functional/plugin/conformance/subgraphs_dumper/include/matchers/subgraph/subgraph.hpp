// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/op/util/op_types.hpp"
#include "common_test_utils/graph_comparator.hpp"

#include "cache/meta/input_info.hpp"
#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"
#include "matchers/single_op/manager.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class SubgraphExtractor {
public:
    // { is_subgraph, model, subgraph, matched_ops{ model_op_name, graph_op_name }}
    using IsSubgraphTuple = std::tuple<bool, std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, std::string>>;
    using Ptr = std::shared_ptr<SubgraphExtractor>;

    SubgraphExtractor() {
        MatchersManager::MatchersMap matchers = {
            { "generic_single_op", SingleOpMatcher::Ptr(new SingleOpMatcher) },
            { "convolutions", ConvolutionsMatcher::Ptr(new ConvolutionsMatcher) },
        };
        m_manager.set_matchers(matchers);
    }

    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref_model) const;
    IsSubgraphTuple is_subgraph(const std::shared_ptr<ov::Model> &model,
                                const std::shared_ptr<ov::Model> &ref_model) const;

    virtual std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model,
                                                bool is_extract_body = true,
                                                bool is_copy_constants = true) {
        return std::list<ExtractedPattern>{};
    };

    void set_extractor_name(const std::string& _extractor_name) { extractor_name = _extractor_name; }

protected:
    std::string extractor_name = "";
    FunctionsComparator comparator = FunctionsComparator::no_default()
        .enable(FunctionsComparator::ATTRIBUTES)
        .enable(FunctionsComparator::NODES)
        .enable(FunctionsComparator::PRECISIONS);
    MatchersManager m_manager = MatchersManager();
    
    inline bool is_node_to_skip(const std::shared_ptr<ov::Node>& node) const {
        return ov::op::util::is_parameter(node) ||
               ov::op::util::is_constant(node) ||
               ov::op::util::is_output(node);
    }
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
