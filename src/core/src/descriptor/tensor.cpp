// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/tensor.hpp"

#include "ngraph/node.hpp"

using namespace std;

ov::descriptor::Tensor::Tensor(const element::Type& element_type,
                               const PartialShape& pshape,
                               const std::unordered_set<std::string>& names)
    : m_element_type(element_type),
      m_partial_shape(pshape),
      m_shape_changed(true) {
    m_name_it = m_names.cend();
    set_names(names);
}

ov::descriptor::Tensor::Tensor(const element::Type& element_type, const PartialShape& pshape, const std::string& name)
    : m_element_type(element_type),
      m_partial_shape(pshape),
      m_shape_changed(true) {
    m_name_it = m_names.cend();
}

ov::descriptor::Tensor::Tensor(const element::Type& element_type,
                               const PartialShape& pshape,
                               ngraph::Node* node,
                               size_t node_output_number)
    : m_element_type(element_type),
      m_partial_shape(pshape),
      m_shape_changed(true) {
    m_name_it = m_names.cend();
}

OPENVINO_SUPPRESS_DEPRECATED_START
void ov::descriptor::Tensor::set_tensor_type(const element::Type& element_type, const PartialShape& pshape) {
    set_element_type(element_type);
    set_partial_shape(pshape);
}

void ov::descriptor::Tensor::set_element_type(const element::Type& element_type) {
    m_element_type = element_type;
}

void ov::descriptor::Tensor::set_partial_shape(const PartialShape& partial_shape) {
    m_partial_shape = partial_shape;
    m_shape_changed = true;
}
OPENVINO_SUPPRESS_DEPRECATED_END

void ov::descriptor::Tensor::invalidate_values() {
    m_upper_value = nullptr;
    m_lower_value = nullptr;
    m_value_label.clear();
}

void ov::descriptor::Tensor::set_lower_value(const ngraph::HostTensorPtr& value) {
    NGRAPH_CHECK(value != nullptr);
    NGRAPH_CHECK(m_partial_shape.same_scheme(value->get_partial_shape()));
    NGRAPH_CHECK(m_element_type == value->get_element_type());
    m_lower_value = value;
}

void ov::descriptor::Tensor::set_upper_value(const ngraph::HostTensorPtr& value) {
    NGRAPH_CHECK(value != nullptr);
    NGRAPH_CHECK(m_partial_shape.same_scheme(value->get_partial_shape()));
    NGRAPH_CHECK(m_element_type == value->get_element_type());
    m_upper_value = value;
}

void ov::descriptor::Tensor::set_value_label(const TensorLabel& value_label) {
    const auto& labels_size = value_label.size();
    if (labels_size == 0) {
        m_value_label.clear();
    } else {
        NGRAPH_CHECK(m_partial_shape.is_static());
        NGRAPH_CHECK(shape_size(m_partial_shape.to_shape()) == labels_size);
        m_value_label = value_label;
    }
}

const ov::Shape& ov::descriptor::Tensor::get_shape() const {
    if (m_partial_shape.is_static()) {
        if (m_shape_changed.load(std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> guard(m_mutex);
            if (m_shape_changed)  // double check after mutex lock
            {
                m_shape = m_partial_shape.to_shape();
                m_shape_changed = false;
            }
        }
        return m_shape;
    } else {
        throw std::invalid_argument("get_shape was called on a descriptor::Tensor with dynamic shape");
    }
}

size_t ov::descriptor::Tensor::size() const {
    const bool bitwidth_less_than_byte = m_element_type.bitwidth() < 8;
    if (bitwidth_less_than_byte) {
        return static_cast<size_t>(ceil((1.0 * shape_size(get_shape()) * m_element_type.bitwidth()) / 8));
    }
    return shape_size(get_shape()) * m_element_type.size();
}

const std::unordered_set<std::string>& ov::descriptor::Tensor::get_names() const {
    return m_names;
}

const std::string& ov::descriptor::Tensor::get_any_name() const {
    if (m_name_it == m_names.cend()) {
        throw ngraph::ngraph_error("Attempt to get a name for a Tensor without names");
    }
    return *m_name_it;
}

void ov::descriptor::Tensor::set_names(const std::unordered_set<std::string>& names) {
    m_names = names;
    m_name_it = m_names.cbegin();
    for (auto it = m_names.cbegin(); it != m_names.cend(); it++) {
        if (*it < *m_name_it)
            // Update any name
            m_name_it = it;
    }
}

void ov::descriptor::Tensor::add_names(const std::unordered_set<std::string>& names) {
    for (const auto& name : names) {
        auto res = m_names.insert(name);
        if (m_name_it == m_names.end() || *res.first < *m_name_it)
            // Update any name
            m_name_it = res.first;
    }
}

std::string ov::descriptor::get_ov_tensor_legacy_name(const ov::descriptor::Tensor& tensor) {
    return tensor.m_legacy_name;
}

void ov::descriptor::set_ov_tensor_legacy_name(ov::descriptor::Tensor& tensor, const std::string& tensor_name) {
    tensor.m_legacy_name = tensor_name;
}

ostream& ov::descriptor::operator<<(ostream& out, const ov::descriptor::Tensor& tensor) {
    std::string names;
    for (const auto& name : tensor.get_names()) {
        if (!names.empty())
            names += ", ";
        names += name;
    }
    NGRAPH_SUPPRESS_DEPRECATED_START
    if (names.empty())
        names = get_ov_tensor_legacy_name(tensor);
    NGRAPH_SUPPRESS_DEPRECATED_END
    out << "Tensor(" << names << ")";
    return out;
}
