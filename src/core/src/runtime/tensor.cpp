// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/tensor.hpp"

#include "ngraph/log.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace std;

const Shape& runtime::Tensor::get_shape() const {
    return m_descriptor->get_shape();
}

const PartialShape& runtime::Tensor::get_partial_shape() const {
    return m_descriptor->get_partial_shape();
}

const element::Type& runtime::Tensor::get_element_type() const {
    return m_descriptor->get_element_type();
}

size_t runtime::Tensor::get_element_count() const {
    return shape_size(m_descriptor->get_shape());
}

size_t runtime::Tensor::get_size_in_bytes() const {
    return m_descriptor->size();
}

const std::string& runtime::Tensor::get_name() const {
    NGRAPH_SUPPRESS_DEPRECATED_START
    return m_descriptor->get_name();
    NGRAPH_SUPPRESS_DEPRECATED_END
}
