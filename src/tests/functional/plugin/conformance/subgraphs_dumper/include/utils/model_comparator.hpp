// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"
#include "matchers/single_op/manager.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class ModelComparator {
public:
    using Ptr = std::shared_ptr<ModelComparator>;
    // { is_match, subgraph, graph, matched_nodes -> {subgraph_op_name, graph_op_name}}
    using IsSubgraphTuple = std::tuple<bool, std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, std::string>>;
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
                                
    void set_match_coefficient(float _match_coefficient);
    void set_shape_strict_match(bool is_shape_strict_match);

protected:
    MatchersManager m_manager = MatchersManager();
    float match_coefficient = 0.9f;
    static std::shared_ptr<ModelComparator> m_instance;

    ModelComparator() {
        MatchersManager::MatchersMap matchers = {
            { "generic_single_op", SingleOpMatcher::Ptr(new SingleOpMatcher) },
            { "convolutions", ConvolutionsMatcher::Ptr(new ConvolutionsMatcher) },
        };
        m_manager.set_matchers(matchers);
    }
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
