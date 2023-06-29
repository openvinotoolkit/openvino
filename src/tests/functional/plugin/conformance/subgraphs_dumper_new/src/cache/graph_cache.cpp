// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache/graph_cache.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

std::shared_ptr<GraphCache> GraphCache::m_cache_instance = nullptr;

void GraphCache::update_cache(const std::shared_ptr<ov::Model>& model, const std::string& model_meta_data, bool extract_body) {}
void GraphCache::serialize_cache() {}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov