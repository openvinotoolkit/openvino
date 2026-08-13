// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gguf_builder_decoder.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov {
namespace frontend {
namespace gguf {

GgufBuilderDecoder::GgufBuilderDecoder(std::shared_ptr<GgufGraph> graph) : m_graph(std::move(graph)) {}

const GgufOp& GgufBuilderDecoder::node() const {
    OPENVINO_ASSERT(m_node_idx >= 0 && static_cast<size_t>(m_node_idx) < m_graph->nodes.size(),
                    "[gguf] node index out of range: ",
                    m_node_idx);
    return m_graph->nodes[m_node_idx];
}

// ---- Per-node typed attribute ----
//
// In addition to keys stored in GgufOp::attributes, the following reserved keys are
// served so external converters can access per-input/output metadata through the public
// base NodeContext::get_attribute<T>() interface without including internal headers:
//
//   "input_shape[N]"       -> ov::PartialShape  for input N (0-based)
//   "input_type[N]"        -> ov::element::Type for input N
//   "input_stride[N]"      -> std::vector<size_t> for input N
//   "input_view_offset[N]" -> int64_t for input N
//   "output_shape"         -> ov::PartialShape of the node output
//   "output_type"          -> ov::element::Type of the node output
//   "rope_config"          -> RopeConfig (model-scope RoPE config; see get_attribute below)

static bool parse_indexed_key(const std::string& name, const std::string& prefix, size_t& out_idx) {
    if (name.size() <= prefix.size() + 2)
        return false;
    if (name.compare(0, prefix.size(), prefix) != 0)
        return false;
    if (name[prefix.size()] != '[' || name.back() != ']')
        return false;
    try {
        out_idx = static_cast<size_t>(std::stoul(name.substr(prefix.size() + 1, name.size() - prefix.size() - 2)));
        return true;
    } catch (...) {
        return false;
    }
}

ov::Any GgufBuilderDecoder::get_attribute(const std::string& name) const {
    // RoPE config is queried at model scope (TranslateSession::preprocess, to build the shared
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

    const auto& n = node();

    // Reserved keys for per-input metadata
    size_t idx = 0;
    if (parse_indexed_key(name, "input_shape", idx)) {
        if (idx < n.input_names.size()) {
            auto it = n.input_shapes.find(n.input_names[idx]);
            if (it != n.input_shapes.end())
                return it->second;
        }
        return {};
    }
    if (parse_indexed_key(name, "input_type", idx)) {
        if (idx < n.input_names.size()) {
            auto it = n.input_types.find(n.input_names[idx]);
            if (it != n.input_types.end())
                return it->second;
        }
        return {};
    }
    if (parse_indexed_key(name, "input_stride", idx)) {
        if (idx < n.input_names.size()) {
            auto it = n.input_strides.find(n.input_names[idx]);
            if (it != n.input_strides.end())
                return it->second;
        }
        return {};
    }
    if (parse_indexed_key(name, "input_view_offset", idx)) {
        if (idx < n.input_names.size()) {
            auto it = n.input_view_offsets.find(n.input_names[idx]);
            return it != n.input_view_offsets.end() ? ov::Any(it->second) : ov::Any(int64_t{0});
        }
        return {};
    }

    // Reserved keys for per-output metadata
    if (name == "output_shape")
        return n.output_shape;
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

PartialShape GgufBuilderDecoder::get_input_shape(const std::string& name) const {
    const auto& m = node().input_shapes;
    auto it = m.find(name);
    OPENVINO_ASSERT(it != m.end(), "[gguf] no input shape for '", name, "'");
    return it->second;
}

int64_t GgufBuilderDecoder::get_input_view_element_offset(const std::string& name) const {
    // The builder does not emit strided VIEW inputs (it materializes slices as explicit ops), so
    // there is no view offset to convert; the stored offsets, when present, are already in
    // elements. Return 0 when the input is not a view.
    const auto& m = node().input_view_offsets;
    auto it = m.find(name);
    return it == m.end() ? 0 : it->second;
}

size_t GgufBuilderDecoder::get_input_size() const {
    return node().input_names.size();
}

std::vector<std::string> GgufBuilderDecoder::get_input_names() const {
    return node().input_names;
}

// ---- Per-node output metadata ----

PartialShape GgufBuilderDecoder::get_output_shape() const {
    return node().output_shape;
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
