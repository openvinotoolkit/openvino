// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gguf_builder_decoder.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov {
namespace frontend {
namespace gguf {

GgufBuilderDecoder::GgufBuilderDecoder(std::shared_ptr<GgufGraph> graph, int node_index)
    : m_graph(std::move(graph)),
      m_node_idx(node_index) {}

const GgufOp& GgufBuilderDecoder::node() const {
    OPENVINO_ASSERT(m_node_idx >= 0 && static_cast<size_t>(m_node_idx) < m_graph->nodes.size(),
                    "[gguf] node index out of range: ",
                    m_node_idx);
    return m_graph->nodes[m_node_idx];
}

// Expose operation parameters through the common decoder interface. Native nodes do not
// carry source shapes; converters infer them from their OpenVINO inputs.

ov::Any GgufBuilderDecoder::get_attribute(const std::string& name) const {
    // RoPE config is queried at model scope (prepare_graph_inputs, to build the shared
    // sin/cos table) and at node scope (each ROPE op's own config). At MODEL scope (no bound node)
    // expose the graph's config with per_op / n_dims==0 encoding "no shared table". At NODE scope
    // fall through to the node's own "rope_config" attribute -- the builder stores a per-node
    // config on each ROPE op (e.g. gemma4 SWA layers use a different freq_base / n_dims), so the
    // node value must win over the graph default.
    if (name == "rope_config" && m_node_idx < 0) {
        RopeConfig cfg = m_graph->rope_config;
        cfg.per_op = m_graph->use_per_op_rope;
        if (!m_graph->has_rope) {
            cfg.n_dims = 0;
        }
        return cfg;
    }

    // Sliding-window length, queried at model scope only (TranslateSession, to record it in
    // rt_info for AdaptToGenAI; see gguf_swa_window_key).
    if (name == "swa_window_size" && m_node_idx < 0) {
        return m_graph->swa_window_size;
    }

    const auto& n = node();

    // Reserved keys for per-output metadata
    if (name == "output_shape")
        return ov::PartialShape::dynamic();
    if (name == "output_type")
        return n.output_type;

    // Per-node op case (the op translators read it via get_attribute<int>("op_case", 0)).
    if (name == "op_case")
        return n.op_case;

    // Named op attributes
    auto it = n.attributes.find(name);
    return it != n.attributes.end() ? it->second : ov::Any{};
}

// ---- Per-input metadata ----

PartialShape GgufBuilderDecoder::get_input_shape(const std::string&) const {
    return ov::PartialShape::dynamic();
}

int64_t GgufBuilderDecoder::get_input_view_element_offset(const std::string&) const {
    return 0;
}

size_t GgufBuilderDecoder::get_input_size() const {
    return node().input_names.size();
}

std::vector<std::string> GgufBuilderDecoder::get_input_names() const {
    return node().input_names;
}

// ---- Per-node output metadata ----

PartialShape GgufBuilderDecoder::get_output_shape() const {
    return ov::PartialShape::dynamic();
}

std::vector<std::string> GgufBuilderDecoder::get_output_names() const {
    return {node().output_name};
}

// ---- Op type / name ----

const std::string& GgufBuilderDecoder::get_op_type() const {
    return node().op_type;
}

const std::string& GgufBuilderDecoder::get_op_name() const {
    return node().name;
}

void GgufBuilderDecoder::visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> node_visitor) const {
    for (size_t i = 0; i < m_graph->nodes.size(); i++) {
        auto per_node = std::make_shared<GgufBuilderDecoder>(*this);
        per_node->m_node_idx = static_cast<int>(i);
        node_visitor(per_node);
    }
}

// ---- Model-level I/O ----

const std::map<std::string, std::shared_ptr<ov::Node>>& GgufBuilderDecoder::get_model_inputs() const {
    return m_graph->model_inputs;
}

const std::map<std::string, std::shared_ptr<ov::Node>>& GgufBuilderDecoder::get_model_extra_inputs() const {
    return m_graph->model_extra_inputs;
}

std::vector<std::string> GgufBuilderDecoder::get_model_output_names() const {
    return m_graph->model_output_names;
}

const std::vector<std::pair<std::string, std::string>>& GgufBuilderDecoder::get_recurrent_states() const {
    return m_graph->recurrent_states;
}

const ov::AnyMap& GgufBuilderDecoder::get_tokenizer_config() const {
    return m_graph->tokenizer_config;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
