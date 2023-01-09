// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/buffer.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

auto normalize_rank(int32_t allocation_rank, const size_t shape_rank) -> int32_t {
    return allocation_rank < 0 ? allocation_rank + shape_rank : allocation_rank;
}

snippets::op::Buffer::Buffer(const Output<Node>& x, const int32_t allocation_rank) : Op({x}), m_allocation_rank(allocation_rank) {
    constructor_validate_and_infer_types();
}

bool snippets::op::Buffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Buffer_visit_attributes);
    visitor.on_attribute("allocation_rank", m_allocation_rank);
    return true;
}

std::shared_ptr<Node> snippets::op::Buffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_buffer = std::make_shared<Buffer>(new_args.at(0), m_allocation_rank);
    return new_buffer;
}

void snippets::op::Buffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    const auto shape_rank = get_input_partial_shape(0).rank();
    if (shape_rank.is_static()) {
        const auto normalized_rank = normalize_rank(m_allocation_rank, shape_rank.get_length());
        NGRAPH_CHECK(normalized_rank >= 0 && normalized_rank <= shape_rank.get_length(),
                     "Buffer has incorrect allocation rank: " + std::to_string(m_allocation_rank));
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

size_t ngraph::snippets::op::Buffer::get_byte_size() const {
    const auto pshape = get_input_partial_shape(0);
    NGRAPH_CHECK(pshape.is_static(), "Buffer should have static shapes for memory allocation");
    const auto shape = pshape.get_shape();
    const auto normalized_rank = normalize_rank(m_allocation_rank, shape.size());
    return ngraph::shape_size(shape.rbegin(), shape.rbegin() + normalized_rank) * get_element_type().size();
}
