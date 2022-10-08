// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/tensor.hpp"

#include "ngraph/node.hpp"

using namespace std;

ov::descriptor::Tensor::Tensor(const element::Type& element_type, const PartialShape& pshape, const std::string& name)
    : m_element_type(element_type),
      m_partial_shape(pshape),
      m_name(name),
      m_shape_changed(true) {}

ov::descriptor::Tensor::Tensor(const element::Type& element_type,
                               const PartialShape& pshape,
                               ngraph::Node* node,
                               size_t node_output_number)
    : m_element_type(element_type),
      m_partial_shape(pshape),
      m_shape_changed(true) {}

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

NGRAPH_SUPPRESS_DEPRECATED_START
void ov::descriptor::Tensor::set_name(const string& name) {
    m_name = name;
}

const std::string& ov::descriptor::Tensor::get_name() const {
    return m_name;
}
NGRAPH_SUPPRESS_DEPRECATED_END

const std::unordered_set<std::string>& ov::descriptor::Tensor::get_names() const {
    return m_names;
}

std::string ov::descriptor::Tensor::get_any_name() const {
    if (m_names.empty()) {
        throw ngraph::ngraph_error("Attempt to get a name for a Tensor without names");
    }
    // As unordered_set for std::string doesn't guaranty the same elements order between runs
    // we have to manually determine the order by sorting tensor name in lexicographical and returning the first one
    std::set<std::string> sorted_names(m_names.begin(), m_names.end());
    return *sorted_names.begin();
}

void ov::descriptor::Tensor::set_names(const std::unordered_set<std::string>& names) {
    m_names = names;
}

void ov::descriptor::Tensor::add_names(const std::unordered_set<std::string>& names) {
    for (const auto& name : names) {
        m_names.insert(name);
    }
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
        names = tensor.get_name();
    NGRAPH_SUPPRESS_DEPRECATED_END
    out << "Tensor(" << names << ")";
    return out;
}
