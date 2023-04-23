// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/buffer.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"


using namespace std;
using namespace ngraph;

auto normalize_rank(int32_t allocation_rank, const size_t shape_rank) -> int32_t {
    return allocation_rank < 0 ? allocation_rank + static_cast<int32_t>(shape_rank) : allocation_rank;
}

snippets::op::Buffer::Buffer(const ov::Shape& shape)
    : Op(), m_type(Type::NewMemory), m_shape(shape) {
    constructor_validate_and_infer_types();
}

snippets::op::Buffer::Buffer(const ov::Output<ov::Node>& arg, const ov::Shape& shape)
    : Op({arg}), m_type(Type::IntermediateMemory), m_shape(shape) {
    constructor_validate_and_infer_types();
}

snippets::op::Buffer::Buffer(const ov::Output<ov::Node>& arg, int32_t allocation_rank)
    : Op({arg}), m_type(Type::IntermediateMemory) {
    const auto pshape = arg.get_partial_shape();
    OPENVINO_ASSERT(pshape.is_static(), "Buffer supports only static input shape");
    const auto shape = pshape.get_shape();
    const auto normalize_rank = utils::normalize_rank(static_cast<int32_t>(allocation_rank), shape.size());
    const auto offset = static_cast<int32_t>(shape.size()) - normalize_rank;
    m_shape = {shape.begin() + offset, shape.end()};
    constructor_validate_and_infer_types();
}

bool snippets::op::Buffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Buffer_visit_attributes);
    visitor.on_attribute("allocation_shape", m_shape);
    return true;
}

void snippets::op::Buffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    ov::element::Type output_type;
    ov::Shape output_shape;
    if (m_type == Type::NewMemory) {
        OPENVINO_ASSERT(get_input_size() == 0, "Buffer with new allocated memory must to not have arguments!");
        output_shape = m_shape;
        output_type = ov::element::u8;  // 1Byte
    } else if (m_type == Type::IntermediateMemory) {
        const auto input_shape = get_input_partial_shape(0);
        OPENVINO_ASSERT(input_shape.is_static(), "Buffer supports only static input shape");
        output_type = get_input_element_type(0);
        output_shape = input_shape.get_shape();
    } else {
        OPENVINO_THROW("Buffer supports only the following types: NewMemory and IntermediateMemory");
    }
    set_output_type(0, output_type, output_shape);
}

std::shared_ptr<Node> snippets::op::Buffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (m_type == Type::NewMemory) {
         return std::make_shared<Buffer>(m_shape);
    } else if (m_type == Type::IntermediateMemory) {
        return std::make_shared<Buffer>(new_args.at(0), m_shape);
    }
    OPENVINO_THROW("Buffer supports only the following types: NewMemory and IntermediateMemory");
}

size_t ngraph::snippets::op::Buffer::get_byte_size() const {
    const auto shape = get_allocation_shape();
    return ngraph::shape_size(shape) * get_element_type().size();
}
