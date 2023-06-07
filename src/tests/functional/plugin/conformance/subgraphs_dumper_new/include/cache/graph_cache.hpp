// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class GraphCache final : public virtual ICache {
public:
    void update_cache(const std::shared_ptr<ov::Model>& model, const std::string& model_meta_data,
                      bool extract_body = true) override;
    void serialize_cache() override;

    static std::shared_ptr<GraphCache>& get() {
        static std::shared_ptr<GraphCache> cache_instance;
        return cache_instance;
    }

private:
    std::map<std::shared_ptr<ov::Model>, MetaInfo> m_graph_cache;
    GraphCache() = default;
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov