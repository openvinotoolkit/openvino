// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "op_conformance_utils/meta_info/input_info.hpp"
#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"
#include "matchers/single_op/manager.hpp"

namespace ov {
namespace util {

class ModelComparator {
public:
    using Ptr = std::shared_ptr<ModelComparator>;
    // { is_match, subgraph, graph, matched_nodes -> {subgraph_op_name, graph_op_name}}
    using IsSubgraphTuple = std::tuple<bool, std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::unordered_map<std::string, std::string>>;
    using InputInfo = ov::conformance::InputInfo;
    // { model, subgraph, graph, subgraph_in_info, model_in_info, }
    using ExtractedSubgraphTuple = std::tuple<bool, std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, InputInfo>, std::map<std::string, InputInfo>>;

    static std::shared_ptr<ModelComparator> get(bool in_is_match_shapes = false) {
        if (m_instance == nullptr) {
            m_instance = std::shared_ptr<ModelComparator>(new ModelComparator);
        }
        return m_instance;
    }

    IsSubgraphTuple is_subgraph(const std::shared_ptr<ov::Model> &model,
                                const std::shared_ptr<ov::Model> &ref_model) const;

    bool match(const std::shared_ptr<ov::Node> &node,
               const std::shared_ptr<ov::Node> &ref_node) const;
    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref_model) const;

    std::pair<bool, std::map<std::string, InputInfo>>
    match(const std::shared_ptr<ov::Model> &model,
          const std::shared_ptr<ov::Model> &ref_model,
          const std::map<std::string, InputInfo> &in_info,
          const std::map<std::string, InputInfo> &in_info_ref);
    ExtractedSubgraphTuple
    is_subgraph(const std::shared_ptr<ov::Model> &model,
                const std::shared_ptr<ov::Model> &ref_model,
                const std::map<std::string, InputInfo> &in_info,
                const std::map<std::string, InputInfo> &in_info_ref);

    // {{matched_node_id}}
    std::vector<std::vector<size_t>>
    get_matched_op_patterns(const ov::NodeVector& ordered_nodes);

    // { op_name_subgraph, op_name_graph}
    std::unordered_map<std::string, std::string>
    get_matched_ops_in_graphs(const std::shared_ptr<ov::Model>& subgraph,
                              const std::shared_ptr<ov::Model>& graph,
                              bool is_check_inputs = false) const;
                                
    void set_match_coefficient(float _match_coefficient);
    float get_match_coefficient() { return match_coefficient; }
    void set_shape_strict_match(bool is_shape_strict_match);
    void set_match_attributes(bool match_attributes);
    void set_match_in_types(bool match_in_types);

protected:
    ov::tools::subgraph_dumper::MatchersManager m_manager = ov::tools::subgraph_dumper::MatchersManager();
    float match_coefficient = 0.9f;
    static std::shared_ptr<ModelComparator> m_instance;

    ModelComparator() {
        ov::tools::subgraph_dumper::MatchersManager::MatchersMap matchers = {
            { "generic_single_op", ov::tools::subgraph_dumper::SingleOpMatcher::Ptr(new ov::tools::subgraph_dumper::SingleOpMatcher) },
            { "convolutions", ov::tools::subgraph_dumper::ConvolutionsMatcher::Ptr(new ov::tools::subgraph_dumper::ConvolutionsMatcher) },
        };
        m_manager.set_matchers(matchers);
    }
};

}  // namespace util
}  // namespace ov
