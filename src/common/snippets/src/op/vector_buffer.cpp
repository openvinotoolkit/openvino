// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/vector_buffer.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

snippets::op::VectorBuffer::VectorBuffer(const ov::element::Type element_type) : Op(), m_element_type(std::move(element_type)) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> snippets::op::VectorBuffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(VectorBuffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<VectorBuffer>(m_element_type);
}

void snippets::op::VectorBuffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(VectorBuffer_validate_and_infer_types);
    set_output_type(0, m_element_type, Shape{1lu});
}
