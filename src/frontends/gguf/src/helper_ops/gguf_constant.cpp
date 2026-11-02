// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_constant.hpp"

#include "openvino/core/except.hpp"
#include "quant/weights.hpp"

namespace ov::frontend::gguf {

GGUFConstant::GGUFConstant(const GgufWeight& source) : m_source(source) {
    OPENVINO_ASSERT(source.data, "GGUF weight has no source storage");
    constructor_validate_and_infer_types();
}

void GGUFConstant::validate_and_infer_types() {
    auto shape = m_source.shape;
    auto type = ov::element::f32;
    if (!m_source.native && m_source.qtype == GGUF_TYPE_MXFP4 && shape.size() > 2) {
        OPENVINO_ASSERT(shape.back() % 32 == 0, "Invalid packed MXFP4 expert width");
        shape = {1, shape[shape.size() - 3], shape[shape.size() - 2], shape.back() / 32, 17};
        type = ov::element::u8;
    }
    set_output_type(0, type, shape);
}

std::shared_ptr<ov::Node> GGUFConstant::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.empty(), "GGUFConstant has no inputs");
    return std::make_shared<GGUFConstant>(m_source);
}

std::shared_ptr<ov::Node> GGUFConstant::materialize() const {
    return make_weight_node(m_source);
}
bool GGUFConstant::is_u2() const {
    return m_source.qtype == GGUF_TYPE_Q2_0 || m_source.qtype == GGUF_TYPE_PQ2_0 || m_source.qtype == GGUF_TYPE_PTQ1_0;
}
GgufTensorType GGUFConstant::qtype() const {
    return m_source.qtype;
}
const GgufWeight& GGUFConstant::source() const {
    return m_source;
}

}  // namespace ov::frontend::gguf
