// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_graph.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/tensor.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

// Split "<something>.weight" into "<something>"; return the name unchanged when it does not end
// in ".weight" (biases and other plain tensors keep their full name as the base).
std::string strip_weight_suffix(const std::string& name);

// Builds OpenVINO values through the shared GGML converters.
class GraphEmitter {
public:
    // `weights` is the parser's tensor table; it is referenced, not copied, and must outlive the emitter.
    GraphEmitter(const GgufWeightTable& weights,
                 std::string arch,
                 const std::unordered_map<std::string, CreatorFunction>* translators = nullptr);

    // ---- graph under construction ----
    const std::shared_ptr<GgufGraph>& graph() const {
        return m_graph;
    }

    bool has_weight(const std::string& name) const {
        return m_weights.count(name) > 0 || m_graph_weights.count(name) > 0;
    }

    const ov::Shape& weight_shape(const std::string& name) const;
    int64_t weight_rows(const std::string& name) const;
    GgufTensorType weight_qtype(const std::string& base) const;
    // Whether selecting `columns` of a parser weight keeps its quantization groups whole.
    bool column_groups_aligned(const std::string& name, const std::vector<int64_t>& columns) const;
    void add_generated_weight(const std::string& name, const ov::Tensor& tensor);

    const ov::Output<ov::Node>& value(const std::string& name) const;

    bool has_model_input(const std::string& name) const {
        return m_graph->model_inputs.count(name) > 0;
    }

    // ---- emission ----

    // Append one op node. `inputs` are producer tensor names (weights / model inputs / earlier node
    // outputs). Returns the output tensor name (== node name).
    std::string add_op(const std::string& op_type,
                       const std::string& name,
                       const std::vector<std::string>& inputs,
                       int op_case = 0,
                       std::map<std::string, ov::Any> attrs = {},
                       std::vector<std::string> extra_outputs = {});

    std::shared_ptr<ov::op::v0::Parameter> add_input(const std::string& name,
                                                     ov::element::Type type,
                                                     const ov::PartialShape& shape);

    void add_extra_input(const std::string& name, int64_t value);

    void add_extra_input_node(const std::string& name, const std::shared_ptr<ov::Node>& node);

    void add_weight(const std::string& ggml_name);
    void add_weight_from(const std::string& node_name, const std::string& src_base);
    // Emit weight `name` computed by `op_type` from weights `sources`. It takes the first source's
    // qtype and Hadamard rotation; all sources must agree on the rotation.
    std::string derive_weight(const std::string& name,
                              const std::string& op_type,
                              const std::vector<std::string>& sources,
                              int op_case,
                              std::map<std::string, ov::Any> attrs = {});

    bool weight_emitted(const std::string& name) const {
        return m_emitted_weights.count(name) > 0;
    }

    void configure_hadamard(const std::map<std::string, GGUFMetaData>& config);
    void validate_hadamard() const;
    bool has_hadamard() const {
        return m_hadamard_block != 0;
    }
    bool has_grouped_hadamard_gdn() const {
        return m_hadamard_v_heads > 0;
    }

private:
    void register_graph_weight(const std::string& name, GgufTensorType qtype);
    struct GraphWeight {
        ov::Shape shape;
        GgufTensorType qtype;
    };
    void emit_weight(const std::string& name, const GgufWeight& source);

    const GgufWeightTable& m_weights;
    std::unordered_map<std::string, GraphWeight> m_graph_weights;
    // Architecture name, used only to make a missing-tensor diagnostic actionable.
    std::string m_arch;

    std::shared_ptr<GgufGraph> m_graph;
    std::shared_ptr<ConstantCache> m_constant_cache = std::make_shared<ConstantCache>();
    std::unordered_map<std::string, CreatorFunction> m_translators;
    // Names of weights already emitted as GGML_OP_NONE leaves, so a weight referenced by several
    // ops (or tied, e.g. MQA tie-V) is emitted once.
    std::set<std::string> m_emitted_weights;
    int64_t m_hadamard_block = 0;
    int64_t m_hadamard_v_heads = 0;
    int64_t m_hadamard_k_heads = 0;
    std::set<std::string> m_hadamard_weights;
    std::set<std::string> m_hadamard_inverse_weights;
    std::set<std::string> m_hadamard_consumed;
    std::map<int64_t, ov::Tensor> m_hadamard_signs;
};

// Shapes are kept in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0] (reverse of GGUF
// on-disk order), matching the decoder's get_shape(). The translators consume them as-is.
inline ov::PartialShape ps(std::vector<int64_t> dims) {
    return ov::PartialShape(std::move(dims));
}

}  // namespace ov::frontend::gguf
