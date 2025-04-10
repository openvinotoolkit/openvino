// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/tensor.hpp"

#include "atomic_guard.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/util/symbolic_info.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace descriptor {

/** @brief Helper class to store Tensor shape information.*/
class ShapeInfo {
public:
    ShapeInfo() = default;
    ShapeInfo(const PartialShape& shape) : m_partial_shape{shape} {}

    void set_partial_shape(PartialShape shape) {
        AtomicGuard lock(m_shape_changing);
        m_partial_shape = std::move(shape);
        m_shape_changed = true;
    }

    const PartialShape& get_partial_shape() const {
        return m_partial_shape;
    }

    const Shape& get_shape() const {
        AtomicGuard lock(m_shape_changing);
        if (m_shape_changed) {
            m_shape = m_partial_shape.to_shape();
            m_shape_changed = false;
        }
        return m_shape;
    }

private:
    PartialShape m_partial_shape{};
    mutable Shape m_shape{};
    mutable std::atomic<bool> m_shape_changing{false};
    mutable bool m_shape_changed{true};
};

// --- Tensor descriptor interface
ITensorDescriptor::~ITensorDescriptor() = default;

/** @brief Basic tensor descriptor. */
class BasicTensor : public ITensorDescriptor {
public:
    BasicTensor() = default;

    BasicTensor(const element::Type& et, const PartialShape& shape, const std::unordered_set<std::string>& names)
        : m_element_type{et},
          m_shape_info{shape},
          m_names{names},
          m_name_it{find_new_any_name(m_names)},
          m_rt_map{} {}

    virtual const element::Type& get_element_type() const override {
        return m_element_type;
    }

    virtual const PartialShape& get_partial_shape() const override {
        return m_shape_info.get_partial_shape();
    }

    virtual const Shape& get_shape() const override {
        return m_shape_info.get_shape();
    }

    virtual void set_type_shape(const element::Type& et, const PartialShape& shape) override {
        m_element_type = et;
        m_shape_info.set_partial_shape(shape);
    }

    void set_names(const std::unordered_set<std::string>& names) override {
        m_names = names;
        m_name_it = find_new_any_name(m_names);
    };

    void add_names(const std::unordered_set<std::string>& names) override {
        m_names.insert(names.begin(), names.end());
        m_name_it = find_new_any_name(m_names);
    }

    const std::unordered_set<std::string>& get_names() const override {
        return m_names;
    }

    const std::unordered_set<std::string>& get_all_names() const override {
        return get_names();
    }

    const std::string& get_any_name() const override {
        OPENVINO_ASSERT(!get_names().empty(), "Attempt to get a name for a Tensor without names");
        return *m_name_it;
    }

    const RTMap& rt_map() const override {
        return m_rt_map;
    }

    RTMap& rt_map() override {
        return m_rt_map;
    };

    size_t pointer_hash() const noexcept override {
        return std::hash<decltype(this)>()(this);
    }

private:
    element::Type m_element_type;
    ShapeInfo m_shape_info;
    std::unordered_set<std::string> m_names;
    std::unordered_set<std::string>::const_iterator m_name_it;
    RTMap m_rt_map;

    static decltype(m_name_it) find_new_any_name(const decltype(m_names)& names) {
        return std::min_element(names.begin(), names.end());
    }
};

// --- TensorExtension
const ITensorDescriptor& TensorExtension::get_descriptor(const Tensor& tensor) {
    return *tensor.m_impl;
}

std::shared_ptr<ITensorDescriptor>& TensorExtension::get_descriptor_ptr(Tensor& tensor) {
    return tensor.m_impl;
}

bool TensorExtension::Equal::operator()(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) const {
    return TensorExtension::get_descriptor(*lhs).pointer_hash() == TensorExtension::get_descriptor(*rhs).pointer_hash();
}

size_t TensorExtension::Hasher::operator()(const std::shared_ptr<Tensor>& tensor) const {
    return get_descriptor(*tensor).pointer_hash();
}

// --- Tensor
Tensor::Tensor(const element::Type& element_type,
               const PartialShape& pshape,
               const std::unordered_set<std::string>& names)
    : m_impl(std::make_shared<BasicTensor>(element_type, pshape, names)) {}

Tensor::Tensor(const element::Type& element_type, const PartialShape& pshape, ov::Node* node, size_t)
    : m_impl(std::make_shared<BasicTensor>(element_type, pshape, std::unordered_set<std::string>{})) {}

void Tensor::invalidate_values() {
    if (ov::skip_invalidation(*this))
        return;
    m_upper_value = {};
    m_lower_value = {};
    m_value_symbol.clear();
}

void Tensor::set_lower_value(const ov::Tensor& value) {
    OPENVINO_ASSERT(static_cast<bool>(value));
    OPENVINO_ASSERT(get_partial_shape().same_scheme(value.get_shape()));
    OPENVINO_ASSERT(get_element_type() == value.get_element_type());
    m_lower_value = value;
}

void Tensor::set_upper_value(const ov::Tensor& value) {
    OPENVINO_ASSERT(static_cast<bool>(value));
    OPENVINO_ASSERT(get_partial_shape().same_scheme(value.get_shape()));
    OPENVINO_ASSERT(get_element_type() == value.get_element_type());
    m_upper_value = value;
}

void Tensor::set_value_symbol(const TensorSymbol& value_symbol) {
    const auto& symbols_size = value_symbol.size();
    if (symbols_size == 0) {
        m_value_symbol.clear();
    } else {
        OPENVINO_ASSERT(get_partial_shape().is_static());
        OPENVINO_ASSERT(shape_size(get_partial_shape().to_shape()) == symbols_size);
        m_value_symbol = value_symbol;
    }
}

const ov::Tensor& Tensor::get_lower_value() const {
    return m_lower_value;
}

const ov::Tensor& Tensor::get_upper_value() const {
    return m_upper_value;
}

TensorSymbol Tensor::get_value_symbol() const {
    return m_value_symbol;
}

bool Tensor::has_and_set_bound() const {
    return m_upper_value && m_lower_value && m_upper_value.data() == m_lower_value.data();
}

const element::Type& Tensor::get_element_type() const {
    return m_impl->get_element_type();
}

const PartialShape& Tensor::get_partial_shape() const {
    return m_impl->get_partial_shape();
}
const Shape& Tensor::get_shape() const {
    return m_impl->get_shape();
}

size_t Tensor::size() const {
    return element::get_memory_size(get_element_type(), shape_size(get_shape()));
}

const std::unordered_set<std::string>& Tensor::get_names() const {
    return m_impl->get_names();
}

const RTMap& Tensor::get_rt_info() const {
    return m_impl->rt_map();
}

RTMap& Tensor::get_rt_info() {
    return m_impl->rt_map();
}

const std::string& Tensor::get_any_name() const {
    return m_impl->get_any_name();
}

void Tensor::set_names(const std::unordered_set<std::string>& names) {
    m_impl->set_names(names);
}

void Tensor::add_names(const std::unordered_set<std::string>& names) {
    m_impl->add_names(names);
}

void Tensor::clone_from(const Tensor& other) {
    m_impl->set_type_shape(other.get_element_type(), other.get_partial_shape());
    set_names(other.get_names());
    m_lower_value = other.get_lower_value();
    m_upper_value = other.get_upper_value();
    m_value_symbol = other.get_value_symbol();
    get_rt_info() = other.get_rt_info();
}

void set_tensor_type(Tensor& tensor, const element::Type& element_type, const PartialShape& pshape) {
    TensorExtension::get_descriptor_ptr(tensor)->set_type_shape(element_type, pshape);
}

void set_element_type(Tensor& tensor, const element::Type& element_type) {
    TensorExtension::get_descriptor_ptr(tensor)->set_type_shape(element_type, tensor.get_partial_shape());
}

void copy_tensor_names(Tensor& dst, const Tensor& src) {
    dst.set_names(TensorExtension::get_descriptor(src).get_all_names());
}

std::ostream& operator<<(std::ostream& out, const Tensor& tensor) {
    out << "Tensor(" << util::join(tensor.get_names()) << ")";
    return out;
}
}  // namespace descriptor
}  // namespace ov
