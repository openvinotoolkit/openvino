// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_constant.hpp"

#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {

GGUFConstant::GGUFConstant(const WeightTensors& tensors, GgufTensorType qtype, const std::string& source_name)
    : m_tensors(tensors),
      m_qtype(qtype),
      m_source_name(source_name),
      m_shape(tensors.weight.get_shape()),
      m_type(ov::element::f32) {
    if (tensors.weight.get_element_type() == ov::element::u32) {
        m_shape.back() *= 8;
    }
    constructor_validate_and_infer_types();
}

GGUFConstant::GGUFConstant(const ov::Shape& shape,
                           const ov::element::Type& type,
                           GgufTensorType qtype,
                           const std::string& source_name,
                           std::function<std::shared_ptr<ov::Node>()> materialize)
    : m_qtype(qtype),
      m_source_name(source_name),
      m_shape(shape),
      m_type(type),
      m_materialize(std::move(materialize)) {
    constructor_validate_and_infer_types();
}

void GGUFConstant::validate_and_infer_types() {
    set_output_type(0, m_type, m_shape);
}

std::shared_ptr<ov::Node> GGUFConstant::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.empty(), "GGUFConstant has no inputs");
    if (m_materialize) {
        return std::make_shared<GGUFConstant>(m_shape, m_type, m_qtype, m_source_name, m_materialize);
    }
    return std::make_shared<GGUFConstant>(m_tensors, m_qtype, m_source_name);
}

std::shared_ptr<ov::Node> GGUFConstant::materialize() const {
    return m_materialize ? m_materialize() : make_weight_node(m_tensors, m_qtype, m_source_name);
}

bool GGUFConstant::is_u2() const {
    return m_qtype == GGUF_TYPE_Q2_0 || m_qtype == GGUF_TYPE_PQ2_0 || m_qtype == GGUF_TYPE_PTQ1_0;
}

bool GGUFConstant::can_concat(const GGUFConstant& other) const {
    return m_tensors.weight && other.m_tensors.weight && m_qtype == other.m_qtype && m_shape.size() == 2 &&
           other.m_shape.size() == 2 && m_shape.back() == other.m_shape.back() &&
           same_weight_conversion_policy(m_source_name, other.m_source_name);
}

const WeightTensors& GGUFConstant::tensors() const {
    return m_tensors;
}

GgufTensorType GGUFConstant::qtype() const {
    return m_qtype;
}

const std::string& GGUFConstant::source_name() const {
    return m_source_name;
}

}  // namespace ov::frontend::gguf
