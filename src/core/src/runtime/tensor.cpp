// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/tensor.hpp"

#include "ngraph/log.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace std;

const Shape& ov::Tensor::get_shape() const {
    return m_descriptor->get_shape();
}

const PartialShape& ov::Tensor::get_partial_shape() const {
    return m_descriptor->get_partial_shape();
}

const element::Type& ov::Tensor::get_element_type() const {
    return m_descriptor->get_element_type();
}

size_t ov::Tensor::get_element_count() const {
    return shape_size(m_descriptor->get_shape());
}

size_t ov::Tensor::get_size_in_bytes() const {
    return m_descriptor->size();
}

const std::string& ov::Tensor::get_name() const {
    NGRAPH_SUPPRESS_DEPRECATED_START
    return m_descriptor->get_name();
    NGRAPH_SUPPRESS_DEPRECATED_END
}
