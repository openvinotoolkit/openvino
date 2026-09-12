// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_emitter.hpp"

#include "gguf_builder_decoder.hpp"
#include "op_table.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

// Split "<something>.weight" into "<something>"; return the name unchanged when it does not end
// in ".weight" (biases and other plain tensors keep their full name as the base).
std::string strip_weight_suffix(const std::string& name) {
    static const std::string suffix = ".weight";
    const bool ends_with_weight =
        name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    return ends_with_weight ? name.substr(0, name.size() - suffix.size()) : name;
}

WeightTensors find_weight_tensors(const std::unordered_map<std::string, ov::Tensor>& weights, const std::string& base) {
    WeightTensors tensors;
    if (auto it = weights.find(base + ".weight"); it != weights.end()) {
        tensors.weight = it->second;
    }
    if (auto it = weights.find(base + ".scales"); it != weights.end()) {
        tensors.scales = it->second;
    }
    if (auto it = weights.find(base + ".zp"); it != weights.end()) {
        tensors.zero_point = it->second;
    }
    return tensors;
}

}  // namespace

GraphEmitter::GraphEmitter(std::unordered_map<std::string, ov::Tensor>& weights,
                           std::unordered_map<std::string, GgufTensorType>& qtypes,
                           std::string arch,
                           const std::unordered_map<std::string, CreatorFunction>* translators)
    : m_weights(weights),
      m_qtypes(qtypes),
      m_arch(std::move(arch)),
      m_graph(std::make_shared<GgufGraph>()),
      m_translators(translators ? *translators : get_supported_ops()) {}

const ov::Tensor& GraphEmitter::weight_tensor(const std::string& name) const {
    auto it = m_weights.find(name);
    OPENVINO_ASSERT(it != m_weights.end(),
                    "[GGUF] model is missing expected weight tensor '",
                    name,
                    "' for architecture '",
                    m_arch,
                    "'");
    return it->second;
}

int64_t GraphEmitter::weight_rows(const std::string& name) const {
    auto it = m_weights.find(name);
    if (it == m_weights.end()) {
        return 1;
    }
    const auto& s = it->second.get_shape();
    return s.empty() ? 1 : static_cast<int64_t>(s[0]);
}

const ov::Output<ov::Node>& GraphEmitter::value(const std::string& name) const {
    const auto it = m_graph->values->find(name);
    OPENVINO_ASSERT(it != m_graph->values->end(), "[GGUF] unknown value '", name, "'");
    return it->second;
}

std::string GraphEmitter::add_op(const std::string& op_type,
                                 const std::string& name,
                                 const std::vector<std::string>& inputs,
                                 int op_case,
                                 std::map<std::string, ov::Any> attrs) {
    GgufOp op;
    op.op_type = op_type;
    op.name = name;
    op.input_names = inputs;
    op.output_name = name;
    op.op_case = op_case;
    op.attributes = std::move(attrs);
    m_graph->nodes.push_back(std::move(op));
    auto decoder = std::make_shared<GgufBuilderDecoder>(m_graph);
    prepare_graph_inputs(*m_graph->values, *decoder);
    translate_node(std::make_shared<GgufBuilderDecoder>(m_graph, static_cast<int>(m_graph->nodes.size() - 1)),
                   m_graph->values,
                   m_translators);
    return name;
}

std::shared_ptr<ov::op::v0::Parameter> GraphEmitter::add_input(const std::string& name,
                                                               ov::element::Type type,
                                                               const ov::PartialShape& shape) {
    auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
    p->set_friendly_name(name);
    p->output(0).set_names({name});
    m_graph->model_inputs[name] = p;
    (*m_graph->values)[name] = p;
    return p;
}

void GraphEmitter::add_extra_input(const std::string& name, int64_t value) {
    auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {value});
    c->set_friendly_name(name);
    m_graph->model_extra_inputs[name] = c;
    (*m_graph->values)[name] = c;
}

void GraphEmitter::add_extra_input_node(const std::string& name, const std::shared_ptr<ov::Node>& node) {
    m_graph->model_extra_inputs[name] = node;
    (*m_graph->values)[name] = node;
}

void GraphEmitter::emit_weight_op(const std::string& node_name, const WeightTensors& tensors, GgufTensorType qtype) {
    if (m_emitted_weights.count(node_name)) {
        return;
    }
    m_emitted_weights.insert(node_name);

    std::map<std::string, ov::Any> attrs{{"gguf_weight", true},
                                         {"gguf_qtype", static_cast<int>(qtype)},
                                         {"gguf.blob.weight", tensors.weight}};
    if (tensors.scales) {
        attrs["gguf.blob.scales"] = tensors.scales;
    }
    if (tensors.zero_point) {
        attrs["gguf.blob.zp"] = tensors.zero_point;
    }
    add_op("GGML_OP_NONE", node_name, {}, 0, std::move(attrs));
}

void GraphEmitter::add_weight(const std::string& ggml_name) {
    if (m_emitted_weights.count(ggml_name)) {
        return;
    }
    const std::string base = strip_weight_suffix(ggml_name);

    auto tensors = find_weight_tensors(m_weights, base);
    GgufTensorType qtype = GGUF_TYPE_F16;
    if (auto it = m_qtypes.find(base + ".qtype"); it != m_qtypes.end()) {
        qtype = it->second;
    }

    emit_weight_op(ggml_name, tensors, qtype);
}

void GraphEmitter::add_weight_from(const std::string& node_name, const std::string& src_base) {
    if (m_emitted_weights.count(node_name)) {
        return;
    }
    auto tensors = find_weight_tensors(m_weights, src_base);
    GgufTensorType qtype = GGUF_TYPE_F16;
    if (auto it = m_qtypes.find(src_base + ".qtype"); it != m_qtypes.end()) {
        qtype = it->second;
    }
    emit_weight_op(node_name, tensors, qtype);
}

void GraphEmitter::add_named_weight(const std::string& ggml_name) {
    if (m_emitted_weights.count(ggml_name)) {
        return;
    }
    auto it = m_weights.find(ggml_name);
    OPENVINO_ASSERT(it != m_weights.end(), "[GGUF] weight not found in gguf: ", ggml_name);
    const ov::Tensor& w = it->second;
    // Map the OV element type back to the ggml float qtype so translate_weight rebuilds a plain
    // Constant of the right precision.
    GgufTensorType qtype = w.get_element_type() == ov::element::f32    ? GGUF_TYPE_F32
                           : w.get_element_type() == ov::element::bf16 ? GGUF_TYPE_BF16
                                                                       : GGUF_TYPE_F16;
    emit_weight_op(ggml_name, {w, {}, {}}, qtype);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
