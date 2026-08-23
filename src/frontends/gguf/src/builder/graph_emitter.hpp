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

namespace ov {
namespace frontend {
namespace gguf {

// Appends nodes to a GgufGraph and tracks the shape/type of every tensor emitted so far.
//
// This is the whole "how do I write a node" layer, and it knows nothing about transformers,
// attention or any architecture: it is the GGUF-frontend counterpart of ggml's ggml_context +
// ggml_build_forward_expand. Everything above it (the block library in blocks/, the per-family
// topology builders in arch/) is expressed purely in terms of these calls, which is what lets a
// non-decoder family (mmproj/vision, audio) reuse it verbatim.
//
// Owns the emit-time bookkeeping the graph itself does not carry:
//   - tensor shapes/types, so add_op can fill each node's per-input shape/type metadata for the
//     translators that query it (MUL_MAT, RESHAPE, VIEW);
//   - the set of already-emitted weight leaves, so a weight referenced by several ops (or tied,
//     e.g. MQA tie-V) is emitted exactly once.
class GraphEmitter {
public:
    // `weights` / `qtypes` are the parser's tensor tables; they are referenced, not copied, and
    // must outlive the emitter.
    GraphEmitter(std::unordered_map<std::string, ov::Tensor>& weights,
                 std::unordered_map<std::string, gguf_tensor_type>& qtypes,
                 std::string arch);

    // ---- graph under construction ----
    const std::shared_ptr<GgufGraph>& graph() const {
        return m_graph;
    }

    // ---- weight-table queries ----
    bool has_weight(const std::string& name) const {
        return m_weights.count(name) > 0;
    }

    // Look up a weight tensor by GGUF name, failing with the tensor NAME if the GGUF is missing it
    // (a bare .at(name) throws std::out_of_range with no context). Used wherever a builder reads a
    // weight's shape to size an op; a missing expected tensor means the file does not match the
    // detected architecture.
    const ov::Tensor& weight_tensor(const std::string& name) const;

    // First extent of a weight's OV shape (its row count), 1 when the weight is absent/scalar.
    int64_t weight_rows(const std::string& name) const;

    std::unordered_map<std::string, ov::Tensor>& weights() {
        return m_weights;
    }
    std::unordered_map<std::string, gguf_tensor_type>& qtypes() {
        return m_qtypes;
    }

    // ---- emitted-tensor queries ----
    const ov::PartialShape& shape_of_tensor(const std::string& name) const;

    // Static shape of an already-emitted tensor, for ops whose translator needs its input's own
    // layout (VIEW op_case 3's "input_ggml_shape", which it uses to restore that layout before
    // slicing). Per-node shapes are static except for the model-input Parameters, so a dynamic dim
    // here means the caller asked about a tensor this cannot describe.
    ov::Shape static_shape_of(const std::string& tensor_name) const;

    // Record a shape/type for a tensor the emitter did not create itself (a model input reused as
    // an op input at a representative token length).
    void set_tensor_meta(const std::string& name, const ov::PartialShape& shape, ov::element::Type type);

    bool has_model_input(const std::string& name) const {
        return m_graph->model_inputs.count(name) > 0;
    }

    // ---- emission ----

    // Append one op node. `inputs` are producer tensor names (weights / model inputs / earlier node
    // outputs). Returns the output tensor name (== node name).
    std::string add_op(const std::string& op_type,
                       const std::string& name,
                       const std::vector<std::string>& inputs,
                       const ov::PartialShape& out_shape,
                       ov::element::Type out_type,
                       int op_case = 0,
                       std::map<std::string, ov::Any> attrs = {});

    std::shared_ptr<ov::op::v0::Parameter> add_input(const std::string& name,
                                                     ov::element::Type type,
                                                     const ov::PartialShape& shape);

    void add_extra_input(const std::string& name, int64_t value);

    void add_extra_input_node(const std::string& name, const std::shared_ptr<ov::Node>& node);

    // Emit a weight as a GGML_OP_NONE leaf node carrying the parser's already-extracted tensors
    // (`<base>.weight` [+ `.scales` [+ `.zp`]] + qtype) as node attributes. translate_weight
    // rebuilds the weights/qtypes maps from these attributes and calls make_weight_node(base,
    // weights, qtypes). Routing weights through GGML_OP_NONE makes that the single weight-loading
    // API, shared with the llama.cpp cgraph decoder (which marks the same leaf with raw ggml bytes
    // instead), and keeps the compressed decompression subgraph -- built lazily in translate_weight
    // during the walk -- rather than materializing an ov::Node eagerly here.
    // `node_name` is the tensor name translators reference (the GGML_OP_NONE output);
    // `extracted` maps "<base>.weight"/".scales"/".zp" -> tensor; `qtype` is the ggml type.
    void emit_weight_op(const std::string& node_name,
                        const std::unordered_map<std::string, ov::Tensor>& extracted,
                        gguf_tensor_type qtype,
                        const ov::PartialShape& shape_4d);

    // `ggml_name` is the full tensor name ending in ".weight" (the name translators reference).
    void add_weight(const std::string& ggml_name);

    // Emit a weight node `node_name` (ends in ".weight") reusing the parser's extracted tensors of
    // another weight `src_base` (base without ".weight"). Used for MQA tie-V, where V shares K's
    // weight tensor; the two GGML_OP_NONE leaves reference the same underlying ov::Tensor blobs
    // (cheap: SharedBuffer views into the parser's single quant buffer).
    void add_weight_from(const std::string& node_name, const std::string& src_base);

    // Emit a plain (non-quantized) weight stored under its full GGUF name, e.g. a bias tensor
    // "blk.N.attn_q.bias" (no ".weight" suffix). It flows through the same GGML_OP_NONE +
    // translate_weight path; make_weight_node treats an F16/F32/BF16 blob as a plain Constant.
    void add_named_weight(const std::string& ggml_name);

    bool weight_emitted(const std::string& name) const {
        return m_emitted_weights.count(name) > 0;
    }

private:
    std::unordered_map<std::string, ov::Tensor>& m_weights;
    std::unordered_map<std::string, gguf_tensor_type>& m_qtypes;
    // Architecture name, used only to make a missing-tensor diagnostic actionable.
    std::string m_arch;

    std::shared_ptr<GgufGraph> m_graph;
    std::map<std::string, ov::PartialShape> m_tensor_shapes;
    std::map<std::string, ov::element::Type> m_tensor_types;
    // Names of weights already emitted as GGML_OP_NONE leaves, so a weight referenced by several
    // ops (or tied, e.g. MQA tie-V) is emitted once.
    std::set<std::string> m_emitted_weights;
};

// Shapes are kept in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0] (reverse of GGUF
// on-disk order), matching the decoder's get_shape(). The translators consume them as-is.
inline ov::PartialShape ps(std::vector<int64_t> dims) {
    return ov::PartialShape(std::move(dims));
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
