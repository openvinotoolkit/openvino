// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_emitter.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"

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
                           std::string arch)
    : m_weights(weights),
      m_qtypes(qtypes),
      m_arch(std::move(arch)),
      m_graph(std::make_shared<GgufGraph>()) {}

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

const ov::PartialShape& GraphEmitter::shape_of_tensor(const std::string& name) const {
    auto it = m_tensor_shapes.find(name);
    OPENVINO_ASSERT(it != m_tensor_shapes.end(), "[GGUF] internal: no shape recorded for '", name, "'");
    return it->second;
}

ov::Shape GraphEmitter::static_shape_of(const std::string& tensor_name) const {
    const auto& shape = shape_of_tensor(tensor_name);
    OPENVINO_ASSERT(shape.is_static(),
                    "[GGUF] internal: shape of '",
                    tensor_name,
                    "' is dynamic (",
                    shape,
                    "), cannot be used as a static input shape");
    return shape.to_shape();
}

void GraphEmitter::set_tensor_meta(const std::string& name, const ov::PartialShape& shape, ov::element::Type type) {
    m_tensor_shapes[name] = shape;
    m_tensor_types[name] = type;
}

std::string GraphEmitter::add_op(const std::string& op_type,
                                 const std::string& name,
                                 const std::vector<std::string>& inputs,
                                 const ov::PartialShape& out_shape,
                                 ov::element::Type out_type,
                                 int op_case,
                                 std::map<std::string, ov::Any> attrs) {
    GgufOp op;
    op.op_type = op_type;
    op.name = name;
    op.input_names = inputs;
    op.output_name = name;
    op.output_shape = out_shape;
    op.output_type = out_type;
    op.op_case = op_case;
    op.attributes = std::move(attrs);
    // Fill per-input shape/type from known producers so translators that query them
    // (MUL_MAT, RESHAPE) get sane values.
    for (const auto& in : inputs) {
        if (auto it = m_tensor_shapes.find(in); it != m_tensor_shapes.end()) {
            op.input_shapes[in] = it->second;
        }
        if (auto it = m_tensor_types.find(in); it != m_tensor_types.end()) {
            op.input_types[in] = it->second;
        }
    }
    m_tensor_shapes[name] = out_shape;
    m_tensor_types[name] = out_type;
    m_graph->nodes.push_back(std::move(op));
    return name;
}

std::shared_ptr<ov::op::v0::Parameter> GraphEmitter::add_input(const std::string& name,
                                                               ov::element::Type type,
                                                               const ov::PartialShape& shape) {
    auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
    p->set_friendly_name(name);
    p->output(0).set_names({name});
    m_graph->model_inputs[name] = p;
    return p;
}

void GraphEmitter::add_extra_input(const std::string& name, int64_t value) {
    auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {value});
    c->set_friendly_name(name);
    m_graph->model_extra_inputs[name] = c;
}

void GraphEmitter::add_extra_input_node(const std::string& name, const std::shared_ptr<ov::Node>& node) {
    m_graph->model_extra_inputs[name] = node;
}

void GraphEmitter::emit_weight_op(const std::string& node_name,
                                  const WeightTensors& tensors,
                                  GgufTensorType qtype,
                                  const ov::PartialShape& shape_4d) {
    if (m_emitted_weights.count(node_name)) {
        return;
    }
    m_emitted_weights.insert(node_name);

    GgufOp op;
    op.op_type = "GGML_OP_NONE";
    op.name = node_name;
    op.output_name = node_name;
    op.output_shape = shape_4d;
    op.output_type = ov::element::f32;
    // translate_weight reads these into a WeightTensors payload:
    // "gguf.blob.<sub>" -> extracted tensor (<sub> in {weight, scales, zp}), and the qtype id.
    op.attributes["gguf_weight"] = true;  // marks this GGML_OP_NONE leaf as a weight
    op.attributes["gguf_qtype"] = static_cast<int>(qtype);
    op.attributes["gguf.blob.weight"] = tensors.weight;
    if (tensors.scales) {
        op.attributes["gguf.blob.scales"] = tensors.scales;
    }
    if (tensors.zero_point) {
        op.attributes["gguf.blob.zp"] = tensors.zero_point;
    }
    m_graph->nodes.push_back(std::move(op));

    m_tensor_shapes[node_name] = shape_4d;
    m_tensor_types[node_name] = ov::element::f32;
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

    // Shape padded to the decoder's rank-4 convention; the extents matter only as the
    // per-input shape/type for translators (MUL_MAT) that index dims [1] and [3].
    int64_t rows = 1, cols = 1;
    if (auto it = m_weights.find(ggml_name); it != m_weights.end()) {
        const auto& s = it->second.get_shape();  // [rows, cols(packed)]
        rows = s.size() >= 1 ? static_cast<int64_t>(s[0]) : 1;
        cols = s.size() >= 2 ? static_cast<int64_t>(s[1]) : 1;
    }
    emit_weight_op(ggml_name, tensors, qtype, ov::PartialShape({1, 1, rows, cols}));
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
    int64_t rows = 1, cols = 1;
    if (auto it = m_weights.find(src_base + ".weight"); it != m_weights.end()) {
        const auto& s = it->second.get_shape();
        rows = s.size() >= 1 ? static_cast<int64_t>(s[0]) : 1;
        cols = s.size() >= 2 ? static_cast<int64_t>(s[1]) : 1;
    }
    emit_weight_op(node_name, tensors, qtype, ov::PartialShape({1, 1, rows, cols}));
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
    const auto& s = w.get_shape();
    int64_t n = s.empty() ? 1 : static_cast<int64_t>(s[0]);
    // 2-D plain weights (qwen35's ssm_conv1d, OV [conv_dim, d_conv]) keep both extents;
    // 1-D ones (biases, norm scales) are the common case and stay a trailing vector.
    const ov::PartialShape shape = s.size() == 2 ? ps({1, 1, n, static_cast<int64_t>(s[1])}) : ps({1, 1, 1, n});
    emit_weight_op(ggml_name, {w, {}, {}}, qtype, shape);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
