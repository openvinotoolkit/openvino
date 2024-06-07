// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/tensor.hpp"

#include "atomic_guard.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/symbolic_info.hpp"

ov::descriptor::Tensor::Tensor(const element::Type& element_type,
                               const PartialShape& pshape,
                               const std::unordered_set<std::string>& names)
    : m_element_type(element_type),
      m_partial_shape(pshape) {
    set_names(names);
}

ov::descriptor::Tensor::Tensor(const element::Type& element_type,
                               const PartialShape& pshape,
                               ov::Node* node,
                               size_t node_output_number)
    : m_element_type(element_type),
      m_partial_shape(pshape) {
    m_name_it = m_names.cend();
}

void ov::descriptor::Tensor::invalidate_values() {
    if (ov::skip_invalidation(*this))
        return;
    m_upper_value = {};
    m_lower_value = {};
    m_value_symbol.clear();
}

void ov::descriptor::Tensor::set_lower_value(const ov::Tensor& value) {
    OPENVINO_ASSERT(static_cast<bool>(value));
    OPENVINO_ASSERT(m_partial_shape.same_scheme(value.get_shape()));
    OPENVINO_ASSERT(m_element_type == value.get_element_type());
    m_lower_value = value;
}

void ov::descriptor::Tensor::set_upper_value(const ov::Tensor& value) {
    OPENVINO_ASSERT(static_cast<bool>(value));
    OPENVINO_ASSERT(m_partial_shape.same_scheme(value.get_shape()));
    OPENVINO_ASSERT(m_element_type == value.get_element_type());
    m_upper_value = value;
}

void ov::descriptor::Tensor::set_value_symbol(const TensorSymbol& value_symbol) {
    const auto& symbols_size = value_symbol.size();
    if (symbols_size == 0) {
        m_value_symbol.clear();
    } else {
        OPENVINO_ASSERT(m_partial_shape.is_static());
        OPENVINO_ASSERT(shape_size(m_partial_shape.to_shape()) == symbols_size);
        m_value_symbol = value_symbol;
    }
}

const ov::Shape& ov::descriptor::Tensor::get_shape() const {
    AtomicGuard lock(m_shape_changing);
    if (m_shape_changed) {
        m_shape = m_partial_shape.to_shape();
        m_shape_changed = false;
    }
    return m_shape;
}

size_t ov::descriptor::Tensor::size() const {
    const bool bitwidth_less_than_byte = m_element_type.bitwidth() < 8;
    return bitwidth_less_than_byte ? (shape_size(get_shape()) * m_element_type.bitwidth() + 7) >> 3
                                   : (shape_size(get_shape()) * m_element_type.size());
}

const std::unordered_set<std::string>& ov::descriptor::Tensor::get_names() const {
    return m_names;
}

const std::string& ov::descriptor::Tensor::get_any_name() const {
    if (m_name_it == m_names.cend()) {
        OPENVINO_THROW("Attempt to get a name for a Tensor without names");
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

void ov::descriptor::Tensor::clone_from(const ov::descriptor::Tensor& old) {
    {
        AtomicGuard lock(m_shape_changing);
        m_partial_shape = old.get_partial_shape();
        m_shape_changed = true;
    }
    set_names(old.get_names());
    m_element_type = old.get_element_type();
    m_lower_value = old.get_lower_value();
    m_upper_value = old.get_upper_value();
    m_value_symbol = old.get_value_symbol();
    m_legacy_name = old.m_legacy_name;
    m_rt_info = old.get_rt_info();
}

std::string ov::descriptor::get_ov_tensor_legacy_name(const ov::descriptor::Tensor& tensor) {
    return tensor.m_legacy_name;
}

void ov::descriptor::set_ov_tensor_legacy_name(ov::descriptor::Tensor& tensor, const std::string& tensor_name) {
    tensor.m_legacy_name = tensor_name;
}

void ov::descriptor::set_tensor_type(ov::descriptor::Tensor& tensor,
                                     const element::Type& element_type,
                                     const PartialShape& pshape) {
    tensor.m_element_type = element_type;
    AtomicGuard lock(tensor.m_shape_changing);
    tensor.m_partial_shape = pshape;
    tensor.m_shape_changed = true;
}

void ov::descriptor::set_element_type(ov::descriptor::Tensor& tensor, const element::Type& element_type) {
    tensor.m_element_type = element_type;
}

std::ostream& ov::descriptor::operator<<(std::ostream& out, const ov::descriptor::Tensor& tensor) {
    std::string names;
    for (const auto& name : tensor.get_names()) {
        if (!names.empty())
            names += ", ";
        names += name;
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (names.empty())
        names = get_ov_tensor_legacy_name(tensor);
    OPENVINO_SUPPRESS_DEPRECATED_END
    out << "Tensor(" << names << ")";
    return out;
}
