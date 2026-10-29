// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_emitter.hpp"

#include <algorithm>

#include "gguf_builder_decoder.hpp"
#include "op_table.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "translate_session.hpp"

namespace ov::frontend::gguf {

namespace {

template <typename T>
const T& required_metadata(const std::map<std::string, GGUFMetaData>& config, const std::string& key) {
    const auto entry = config.find(key);
    OPENVINO_ASSERT(entry != config.end(), "[GGUF] missing ", key);
    const auto* value = std::get_if<T>(&entry->second);
    OPENVINO_ASSERT(value, "[GGUF] invalid type for ", key);
    return *value;
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

std::string strip_weight_suffix(const std::string& name) {
    static const std::string suffix = ".weight";
    const bool ends_with_weight =
        name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    return ends_with_weight ? name.substr(0, name.size() - suffix.size()) : name;
}

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
                                 std::map<std::string, ov::Any> attrs,
                                 std::vector<std::string> extra_outputs) {
    if (!inputs.empty() && ((op_type == "GGML_OP_MUL_MAT" && m_hadamard_weights.count(inputs[0])) ||
                            (op_type == "GGML_OP_GET_ROWS" && m_hadamard_inverse_weights.count(inputs[0])))) {
        m_hadamard_consumed.insert(inputs[0]);
        const auto width = value(inputs[0]).get_partial_shape().rbegin()->get_length();
        attrs["hadamard_block"] = m_hadamard_block;
        attrs["hadamard_inverse"] = op_type == "GGML_OP_GET_ROWS";
        if (!m_hadamard_signs.empty()) {
            OPENVINO_ASSERT(m_hadamard_signs.count(width), "[GGUF] missing Hadamard signs for width ", width);
            attrs["hadamard_signs"] = m_hadamard_signs.at(width);
        }
        if (op_type == "GGML_OP_MUL_MAT" && m_hadamard_v_heads && inputs[0].find(".ssm_out.") != std::string::npos) {
            OPENVINO_ASSERT(width % m_hadamard_v_heads == 0, "[GGUF] invalid Hadamard GDN width");
            attrs["hadamard_gdn_heads"] = std::vector<int64_t>{m_hadamard_v_heads / m_hadamard_k_heads,
                                                               m_hadamard_k_heads,
                                                               width / m_hadamard_v_heads};
        }
    }
    GgufOp op;
    op.op_type = op_type;
    op.name = name;
    op.input_names = inputs;
    op.output_name = name;
    op.extra_output_names = std::move(extra_outputs);
    op.op_case = op_case;
    op.attributes = std::move(attrs);
    m_graph->nodes.push_back(std::move(op));
    auto decoder = std::make_shared<GgufBuilderDecoder>(m_graph);
    prepare_graph_inputs(*m_graph->values, *decoder);
    translate_node(std::make_shared<GgufBuilderDecoder>(m_graph, static_cast<int>(m_graph->nodes.size() - 1)),
                   m_graph->values,
                   m_translators,
                   m_constant_cache);
    return name;
}

void GraphEmitter::inherit_hadamard(const std::string& name, const std::string& source) {
    if (m_hadamard_weights.count(source)) {
        m_hadamard_weights.insert(name);
        m_hadamard_consumed.insert(source);
    }
}

void GraphEmitter::validate_hadamard() const {
    for (const auto& names : {m_hadamard_weights, m_hadamard_inverse_weights})
        for (const auto& name : names)
            OPENVINO_ASSERT(m_hadamard_consumed.count(name), "[GGUF] unused Hadamard weight: ", name);
}

void GraphEmitter::configure_hadamard(const std::map<std::string, GGUFMetaData>& config) {
    const std::string prefix = "prism.hadamard.";
    if (!config.count(prefix + "version")) {
        OPENVINO_ASSERT(std::none_of(config.begin(),
                                     config.end(),
                                     [&](const auto& entry) {
                                         return entry.first.compare(0, prefix.size(), prefix) == 0;
                                     }),
                        "[GGUF] missing prism.hadamard.version");
        return;
    }
    const auto scalar = [&](const std::string& key) {
        const auto& t = required_metadata<ov::Tensor>(config, key);
        OPENVINO_ASSERT(t.get_element_type() == ov::element::u32 && t.get_size() == 1, "[GGUF] invalid ", key);
        return *t.data<const uint32_t>();
    };
    const auto flag = [&](const std::string& key) {
        if (!config.count(key))
            return false;
        const auto& t = required_metadata<ov::Tensor>(config, key);
        OPENVINO_ASSERT(t.get_element_type() == ov::element::boolean && t.get_size() == 1, "[GGUF] invalid ", key);
        return *t.data<const char>() != 0;
    };
    const auto version = scalar(prefix + "version");
    const bool tied = flag(prefix + "tied_output");
    OPENVINO_ASSERT((version == 1 || version == 2) && (version == 2) == tied,
                    "[GGUF] inconsistent Hadamard version/tied_output");
    m_hadamard_block = scalar(prefix + "block_size");
    OPENVINO_ASSERT(
        m_hadamard_block > 0 && m_hadamard_block <= 4096 && (m_hadamard_block & (m_hadamard_block - 1)) == 0,
        "[GGUF] invalid Hadamard block size");
    OPENVINO_ASSERT(
        required_metadata<std::string>(config, prefix + "transform") == "normalized-sylvester-walsh-hadamard" &&
            required_metadata<std::string>(config, prefix + "axis") == "input-last-dimension",
        "[GGUF] unsupported Hadamard transform/axis");
    const auto& names = required_metadata<std::vector<std::string>>(config, prefix + "weight_names");
    OPENVINO_ASSERT(!names.empty(), "[GGUF] empty Hadamard weight_names");
    for (const auto& name : names) {
        OPENVINO_ASSERT(has_weight(name) && m_hadamard_weights.insert(name).second,
                        "[GGUF] missing or duplicate Hadamard weight: ",
                        name);
    }
    if (config.count(prefix + "inverse_weight_names")) {
        for (const auto& name : required_metadata<std::vector<std::string>>(config, prefix + "inverse_weight_names")) {
            OPENVINO_ASSERT(name == "token_embd.weight" && has_weight(name) && !m_hadamard_weights.count(name) &&
                                m_hadamard_inverse_weights.insert(name).second,
                            "[GGUF] invalid Hadamard inverse weight: ",
                            name);
        }
    }
    if (tied) {
        OPENVINO_ASSERT(!has_weight("output.weight") && m_hadamard_inverse_weights.count("token_embd.weight"),
                        "[GGUF] tied Hadamard output requires a latent embedding and no output.weight");
        m_hadamard_weights.insert("token_embd.weight");
    } else {
        OPENVINO_ASSERT(m_hadamard_inverse_weights.empty() || has_weight("output.weight"),
                        "[GGUF] latent tied embedding requires Hadamard version 2");
    }
    const auto& mode = required_metadata<std::string>(config, prefix + "sign_mode");
    OPENVINO_ASSERT(mode == "identity" || mode == "explicit", "[GGUF] unsupported Hadamard sign_mode");
    if (mode == "explicit") {
        const auto& widths = required_metadata<ov::Tensor>(config, prefix + "sign_widths");
        const auto& values = required_metadata<ov::Tensor>(config, prefix + "sign_values");
        OPENVINO_ASSERT(widths.get_element_type() == ov::element::i32 && widths.get_size() > 0,
                        "[GGUF] invalid ",
                        prefix + "sign_widths");
        OPENVINO_ASSERT(values.get_element_type() == ov::element::i32, "[GGUF] invalid ", prefix + "sign_values");
        size_t offset = 0;
        for (size_t i = 0; i < widths.get_size(); ++i) {
            const int64_t width = widths.data<const int32_t>()[i];
            OPENVINO_ASSERT(width > 0 && width % m_hadamard_block == 0 &&
                                static_cast<size_t>(width) <= values.get_size() - offset &&
                                !m_hadamard_signs.count(width),
                            "[GGUF] invalid Hadamard sign width");
            ov::Tensor signs(ov::element::f32, {static_cast<size_t>(width)});
            for (int64_t j = 0; j < width; ++j) {
                const auto sign = values.data<const int32_t>()[offset++];
                OPENVINO_ASSERT(sign == -1 || sign == 1, "[GGUF] Hadamard signs must be +/-1");
                signs.data<float>()[j] = static_cast<float>(sign);
            }
            m_hadamard_signs.emplace(width, signs);
        }
        OPENVINO_ASSERT(offset == values.get_size(), "[GGUF] Hadamard sign_values length mismatch");
    }
    for (const auto& names : {m_hadamard_weights, m_hadamard_inverse_weights})
        for (const auto& name : names) {
            const auto& shape = weight_tensor(name).get_shape();
            OPENVINO_ASSERT(shape.size() == 2, "[GGUF] Hadamard requires matrix weights: ", name);
            const auto width = shape.back();
            OPENVINO_ASSERT(width > 0 && width % m_hadamard_block == 0 &&
                                (m_hadamard_signs.empty() || m_hadamard_signs.count(width)),
                            "[GGUF] invalid Hadamard width for ",
                            name);
        }
    if (flag(prefix + "gdn_v_grouped")) {
        m_hadamard_v_heads = required_metadata<int>(config, "ssm_time_step_rank");
        m_hadamard_k_heads = required_metadata<int>(config, "ssm_group_count");
        OPENVINO_ASSERT(
            m_hadamard_k_heads > 0 && m_hadamard_v_heads > 0 && m_hadamard_v_heads % m_hadamard_k_heads == 0,
            "[GGUF] invalid Hadamard GDN heads");
    }
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

WeightTensors GraphEmitter::weight_parts(const std::string& base) const {
    return find_weight_tensors(m_weights, base);
}

GgufTensorType GraphEmitter::weight_qtype(const std::string& base) const {
    return lookup_qtype(base, m_qtypes);
}

void GraphEmitter::add_weight(const std::string& ggml_name) {
    if (m_emitted_weights.count(ggml_name)) {
        return;
    }
    const std::string base = strip_weight_suffix(ggml_name);
    emit_weight_op(ggml_name, weight_parts(base), weight_qtype(base));
}

void GraphEmitter::add_weight_from(const std::string& node_name, const std::string& src_base) {
    if (m_emitted_weights.count(node_name)) {
        return;
    }
    emit_weight_op(node_name, weight_parts(src_base), weight_qtype(src_base));
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

}  // namespace ov::frontend::gguf
