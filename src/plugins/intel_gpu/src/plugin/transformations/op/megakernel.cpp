// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/megakernel.hpp"

namespace ov::intel_gpu::op {

MegaKernel::MegaKernel(const ov::OutputVector& inputs, const MegaKernelAttrs& attrs)
    : Op(inputs),
      m_attrs(attrs) {
    validate_and_infer_types();
}

bool MegaKernel::visit_attributes(ov::AttributeVisitor& visitor) {
    // Flattened to visitor-supported types; output_shapes holds, per output, its rank followed by
    // (input, axis, value) per dimension.
    ov::element::TypeVector output_types;
    std::vector<int64_t> output_shapes;
    for (const auto& out : m_attrs.outputs) {
        output_types.push_back(out.type);
        output_shapes.push_back(static_cast<int64_t>(out.shape.size()));
        for (const auto& d : out.shape)
            output_shapes.insert(output_shapes.end(), {d.input, d.axis, d.value});
    }
    visitor.on_attribute("output_types", output_types);
    visitor.on_attribute("output_shapes", output_shapes);
    visitor.on_attribute("params", m_attrs.params);

    // Rebuild in case the visitor wrote the attributes (deserialization).
    m_attrs.outputs.clear();
    size_t pos = 0;
    for (const auto& type : output_types) {
        OPENVINO_ASSERT(pos < output_shapes.size(), "[MegaKernel] malformed output_shapes attribute");
        const auto rank = static_cast<size_t>(output_shapes[pos++]);
        OPENVINO_ASSERT(pos + 3 * rank <= output_shapes.size(), "[MegaKernel] malformed output_shapes attribute");
        MegaKernelOutput out{type, {}};
        for (size_t i = 0; i < rank; ++i, pos += 3)
            out.shape.push_back({output_shapes[pos], output_shapes[pos + 1], output_shapes[pos + 2]});
        m_attrs.outputs.push_back(std::move(out));
    }
    OPENVINO_ASSERT(pos == output_shapes.size(), "[MegaKernel] malformed output_shapes attribute");
    return true;
}

void MegaKernel::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, !m_attrs.outputs.empty(), "MegaKernel must have at least one output");
    std::vector<ov::PartialShape> input_shapes;
    for (size_t i = 0; i < get_input_size(); ++i)
        input_shapes.push_back(get_input_partial_shape(i));
    set_output_size(m_attrs.outputs.size());
    for (size_t i = 0; i < m_attrs.outputs.size(); ++i)
        set_output_type(i, m_attrs.outputs[i].type, megakernel_output_shape(m_attrs.outputs[i], input_shapes));
}

std::shared_ptr<ov::Node> MegaKernel::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MegaKernel>(new_args, m_attrs);
}

}  // namespace ov::intel_gpu::op
