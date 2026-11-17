// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gguf_constant.hpp"

#include "openvino/core/except.hpp"
#include "quant/weights.hpp"

namespace ov::frontend::gguf {

GGUFConstant::GGUFConstant(const GgufWeight& source) : ov::op::util::FrameworkNode({}, 1), m_source(source) {
    OPENVINO_ASSERT(source.data, "GGUF weight has no source storage");
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs.set_type_name("GGUFConstant");
    attrs.set_opset_name("gguf");
    set_attrs(attrs);
    constructor_validate_and_infer_types();
}

void GGUFConstant::validate_and_infer_types() {
    auto shape = m_source.shape;
    auto type = ov::element::f32;
    if (!m_source.native && m_source.qtype == GGUF_TYPE_MXFP4 && shape.size() > 2) {
        shape = mxfp4_packed_shape(shape);
        type = ov::element::u8;
    }
    set_output_type(0, type, shape);
}

std::shared_ptr<ov::Node> GGUFConstant::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.empty(), "GGUFConstant has no inputs");
    auto clone = std::make_shared<GGUFConstant>(m_source);
    clone->set_attrs(get_attrs());
    return clone;
}

std::shared_ptr<ov::Node> GGUFConstant::materialize() const {
    return make_weight_node(m_source);
}
GgufTensorType GGUFConstant::qtype() const {
    return m_source.qtype;
}
const GgufWeight& GGUFConstant::source() const {
    return m_source;
}

}  // namespace ov::frontend::gguf
