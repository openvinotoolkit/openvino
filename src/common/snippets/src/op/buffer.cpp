// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/op/buffer.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace op {

Buffer::Buffer(const ov::Output<ov::Node>& arg) : Buffer(ov::OutputVector{arg}) {}

Buffer::Buffer(const OutputVector& arguments) : Op(arguments), m_type(Type::IntermediateMemory) {
    constructor_validate_and_infer_types();
}

Buffer::Buffer(const ov::Shape& shape, ov::element::Type element_type) : Op(), m_type(Type::NewMemory), m_output_shape(shape), m_element_type(element_type) {
    constructor_validate_and_infer_types();
}

bool Buffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Buffer_visit_attributes);
    auto shape = utils::pshape_to_vdims(get_output_partial_shape(0));
    auto etype = get_output_element_type(0);
    visitor.on_attribute("shape", shape);
    visitor.on_attribute("element_type", etype);
    return true;
}

void Buffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    if (m_type == Type::NewMemory) {
        OPENVINO_ASSERT(get_input_size() == 0, "NewMemory Buffer mustn't have inputs");
        set_output_type(0, m_element_type, m_output_shape);
    } else if (m_type == Type::IntermediateMemory) {
        OPENVINO_ASSERT(get_input_size() != 0, "IntermediateMemory Buffer must have inputs");
        const auto inputs = input_values();
        const auto inshape = get_input_partial_shape(0);
        const auto intype = get_input_element_type(0);
        OPENVINO_ASSERT(std::all_of(inputs.cbegin() + 1, inputs.cend(),
                                    [&](const ov::Output<ov::Node>& in) { return in.get_partial_shape() == inshape && in.get_element_type() == intype; }),
                        "All inputs of Buffers must have the same shape and element type");
        set_output_type(0, intype, inshape);
    } else {
        OPENVINO_THROW("Unknown Buffer type");
    }
}

std::shared_ptr<Node> Buffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    if (m_type == Type::NewMemory) {
        OPENVINO_ASSERT(new_args.empty(), "NewMemory Buffer mustn't have inputs");
        return std::make_shared<Buffer>(m_output_shape, m_element_type);
    } else if (m_type == Type::IntermediateMemory) {
        return std::make_shared<Buffer>(new_args);
    } else {
        OPENVINO_THROW("Unknown Buffer type");
    }
}

size_t Buffer::get_allocation_size() const {
    if (m_type == Type::NewMemory) {
        const auto pshape = get_output_partial_shape(0);
        OPENVINO_ASSERT(pshape.is_static(), "If Buffer doesn't have source - output shape must be static");
        return ov::shape_size(pshape.get_shape());
    }
    return utils::get_dynamic_value<size_t>();
}

Buffer::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& buffer = ov::as_type_ptr<Buffer>(n);
    OPENVINO_ASSERT(buffer, "Got invalid node in Buffer::ShapeInfer");
    m_type = buffer->m_type;
    OPENVINO_ASSERT(utils::one_of(m_type, Type::IntermediateMemory, Type::NewMemory), "Got invalid Buffer type");
    if (m_type == Type::NewMemory)
        m_shape = buffer->m_output_shape;
}

IShapeInferSnippets::Result Buffer::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    if (m_type == Type::NewMemory) {
        OPENVINO_ASSERT(input_shapes.empty(), "NewMemoryBuffer shape inference mustn't have input shapes");
        return {{m_shape}, ShapeInferStatus::success};
    } else if (m_type == Type::IntermediateMemory) {
        OPENVINO_ASSERT(!input_shapes.empty(), "IntermediateMemoryBuffer shape inference must have input shapes");
        return {{input_shapes[0].get()}, ShapeInferStatus::success};
    }
    OPENVINO_THROW("Uknown Buffer type!");
}

} // namespace op
} // namespace snippets
} // namespace ov
