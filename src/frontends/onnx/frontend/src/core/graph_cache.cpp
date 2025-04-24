// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/graph_cache.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace frontend {
namespace onnx {
void GraphCache::emplace_node(const std::string& name, ov::Output<ov::Node>&& node) {
    m_graph_cache_map[name] = std::move(node);
}

void GraphCache::remove_node(const std::string& name) {
    auto it = m_graph_cache_map.find(name);
    if (it != m_graph_cache_map.end()) {
        m_graph_cache_map.erase(it);
    }
}

ov::Output<ov::Node> GraphCache::get_node(const std::string& name) const {
    try {
        return m_graph_cache_map.at(name);
    } catch (const std::out_of_range&) {
        OPENVINO_THROW(name + " node not found in graph cache");
    }
}

bool GraphCache::contains(const std::string& name) const {
    return (m_graph_cache_map.count(name) > 0);
}
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
