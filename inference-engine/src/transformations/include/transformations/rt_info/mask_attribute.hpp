// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines primitives priority attribute
 * @file primitives_priority_attribute.hpp
 */

#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {


/**
 * @ingroup ie_runtime_attr_api
 * @brief describes dimension values that contains only zeros and can be removed.
 */
class TRANSFORMATIONS_API MaskValue : public std::set<uint64_t>,
                                      public std::enable_shared_from_this<MaskValue> {
public:
    using Ptr = std::shared_ptr<MaskValue>;

    MaskValue() = default;

    MaskValue(std::initializer_list<value_type> list) noexcept
        : std::set<value_type>(list) {
    }

    void add_parent(const MaskValue::Ptr & parent) {
        m_parents.push_back(parent);
        parent->add_consumer(this->shared_from_this());
    }

    void update_dependencies() {
        for (auto & parent : m_parents) {
            if (*parent == *this) {
                continue;
            }
            // TODO: check that new dimension values are in range of existing values
            parent->clear();
            for (auto & it  : *this) {
                parent->insert(it);
            }
            parent->update_dependencies();
        }

        for (auto & consumer : m_consumers) {
            if (*consumer == *this) {
                continue;
            }
            consumer->clear();
            for (auto it = begin(); it != end(); ++it) {
                consumer->insert(*it);
            }
            consumer->update_dependencies();
        }
    }

private:
    void add_consumer(MaskValue::Ptr consumer) {
        m_consumers.emplace_back(std::move(consumer));
    }

    std::vector<MaskValue::Ptr> m_parents;
    std::vector<MaskValue::Ptr> m_consumers;
};

/**
 * @ingroup ie_runtime_attr_api
 * @brief each element in vector represents dimension and each element
 * in set is an id of dimensions which contains zeros.
 */
class TRANSFORMATIONS_API Mask : public std::vector<MaskValue::Ptr> {
public:
    using Ptr = std::shared_ptr<Mask>;

    Mask() = default;

    explicit Mask(const ngraph::PartialShape & shape)
            : std::vector<value_type>() {
        size_t count = shape.rank().get_length();
        for (size_t i = 0; i < count; i++) {
            push_back(std::make_shared<MaskValue>());
        }
    }

    explicit Mask(const size_t & size)
            : std::vector<value_type>() {
        for (size_t i = 0; i < size; i++) {
            push_back(std::make_shared<MaskValue>());
        }
    }

    Mask(std::initializer_list<std::initializer_list<MaskValue::value_type>> list) noexcept
            : std::vector<value_type>() {
        for (const auto & dim_values : list) {
            push_back(std::make_shared<MaskValue>(dim_values));
        }
    }

    void update_dependencies() {
        for (auto it = begin(); it != end(); ++it) {
            it->get()->update_dependencies();
        }
    }
    void invalidate() {
        for (auto it = begin(); it != end(); ++it) {
            it->get()->clear();
        }
        update_dependencies();
    }

    bool all_dims_are_empty() const {
        return std::all_of(begin(), end(),
                           [](const value_type & value) {
                               return (!value || value->empty());
                           });
    }

    bool is_shape_like() const { return m_is_shape_like; }

    void set_shape_like(bool flag) { m_is_shape_like = flag; }

private:
    bool m_is_shape_like{false};
};

TRANSFORMATIONS_API std::ostream & operator<< (std::ostream & out, const Mask & mask);

extern template class TRANSFORMATIONS_API VariantImpl<Mask::Ptr>;

template<>
class TRANSFORMATIONS_API VariantWrapper<Mask::Ptr> : public VariantImpl<Mask::Ptr> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::Mask", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    static std::shared_ptr<VariantWrapper<Mask::Ptr>> create(const value_type & value) {
        return std::make_shared<VariantWrapper<Mask::Ptr>>(value);
    }

    explicit VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}
};

TRANSFORMATIONS_API Mask::Ptr getMask(Output<Node> output);

TRANSFORMATIONS_API void setMask(Output<Node> output, const Mask::Ptr & mask);

}  // namespace ngraph
