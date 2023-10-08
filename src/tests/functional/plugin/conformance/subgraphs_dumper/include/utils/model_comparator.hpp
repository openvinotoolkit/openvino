// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/graph_comparator.hpp"

#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"
#include "matchers/single_op/manager.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class ModelComparator {
public:
    using Ptr = std::shared_ptr<ModelComparator>;
    using IsSubgraphTuple = std::tuple<bool, std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, std::string>>;

    static std::shared_ptr<ModelComparator> get() {
        if (m_instance == nullptr) {
            m_instance = std::shared_ptr<ModelComparator>(new ModelComparator);
        }
        return std::shared_ptr<ModelComparator>(m_instance);
    }

    std::map<std::string, InputInfo>
    align_input_info(const std::shared_ptr<ov::Model>& model,
                     const std::shared_ptr<ov::Model>& model_ref,
                     const std::map<std::string, InputInfo> &in_info,
                     const std::map<std::string, InputInfo> &in_info_ref,
                     const std::map<std::string, std::string> &matched_op = {});

    bool match(const std::shared_ptr<ov::Node> &node,
               const std::shared_ptr<ov::Node> &ref_node);
    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref_model,
               std::map<std::string, InputInfo> &in_info,
               const std::map<std::string, InputInfo> &in_info_ref);

    IsSubgraphTuple is_subgraph(const std::shared_ptr<ov::Model> &model,
                                const std::shared_ptr<ov::Model> &ref_model) const;
                                
    void set_match_coefficient(float _match_coefficient) {
        if (_match_coefficient  < 0 || _match_coefficient > 1) {
            throw std::runtime_error("[ ERROR ] Match coefficient should be from 0 to 1!");
        }
        match_coefficient = _match_coefficient; 
    }

private:
    FunctionsComparator m_comparator = FunctionsComparator::no_default()
        .enable(FunctionsComparator::ATTRIBUTES)
        .enable(FunctionsComparator::NODES)
        .enable(FunctionsComparator::PRECISIONS);
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
    

    bool match(const std::shared_ptr<ov::Model> &model,
               const std::shared_ptr<ov::Model> &ref_model) const;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
