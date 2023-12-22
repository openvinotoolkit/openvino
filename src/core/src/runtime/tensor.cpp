// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/tensor.hpp"

using namespace ngraph;
using namespace std;

OPENVINO_SUPPRESS_DEPRECATED_START

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
