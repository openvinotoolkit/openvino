// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/cache.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class OpCache : public virtual ICache {
public:
    void update_cache(const std::shared_ptr<ov::Model>& model,
                      const std::string& model_path, bool extract_body) override;
    void serialize_cache() override;

    static std::shared_ptr<OpCache> get() {
        if (m_cache_instance == nullptr) {
            m_cache_instance = std::shared_ptr<OpCache>(new OpCache);
        }
        return std::shared_ptr<OpCache>(m_cache_instance);
    }

protected:
    std::map<std::shared_ptr<ov::Node>, MetaInfo> m_ops_cache;
    static std::shared_ptr<OpCache> m_cache_instance;

    OpCache() {
        // SubgraphsDumper::MatchersManager::MatchersMap matchers = {
        //         { "generic_single_op", std::make_shared<SubgraphsDumper::SingleOpMatcher>() },
        //         { "convolutions", std::make_shared<SubgraphsDumper::ConvolutionsMatcher>() },
        //     };
        // m_manager.set_matchers(matchers);
    }

    void update_cache(const std::shared_ptr<ov::Node>& node, const std::string& model_path);
    bool serialize_op(const std::pair<std::shared_ptr<ov::Node>, MetaInfo>& op_info);
    std::string get_rel_serilization_dir(const std::shared_ptr<ov::Node>& node);
    std::shared_ptr<ov::Model> generate_graph_by_node(const std::shared_ptr<ov::Node>& node);
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov