// Copyright (C) 2018-2021 Intel Corporation
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

namespace ngraph {

/**
 * @ingroup ie_runtime_attr_api
 * @brief each element in vector represents dimension and each element
 * in set is an id of dimensions which contains zeros.
 */
class Mask : public std::vector<std::set<uint64_t>>,
             public std::enable_shared_from_this<Mask> {
public:
    using Ptr = std::shared_ptr<Mask>;

    Mask() = default;

    explicit Mask(const ngraph::PartialShape & shape)
            : std::vector<value_type>(shape.rank().get_length()) {
    }

    explicit Mask(const size_t & size)
            : std::vector<value_type>(size) {
    }

    Mask(std::initializer_list<std::initializer_list<uint64_t>> list)
            : std::vector<value_type>() {
        for (const auto & dim_values : list) {
            push_back(dim_values);
        }
    }

    bool all_dims_are_empty() const {
        return std::all_of(begin(), end(),
                           [](const value_type & value) {
                               return value.empty();
                           });
    }

    bool is_shape_like() const { return m_is_shape_like; }

    void set_shape_like(bool flag) { m_is_shape_like = flag; }

    void add_callback(const std::function<bool(Mask::Ptr)> & receive_callback, Mask::Ptr mask) {
        m_callbacks[mask.get()] = receive_callback;
        m_dependencies.push_back(mask.get());
    }

    bool apply_callback(Mask::Ptr mask) {
        // TODO: in case if callback returns false we need to propagate original value
        const auto & ref_state = Mask(*this);
        if (!m_callbacks.at(mask.get())(shared_from_this())) {
            return false;
        }

        if (!m_need_initialization && *this == ref_state) {
            return true;
        }

        m_need_initialization = false;

        for (const auto & m_dependency : m_dependencies) {
            if (!m_dependency->apply_callback(shared_from_this())) {
                return false;
            }
        }
        return true;
    }

    void invalidate() {
        clean_dim_values();
        for (const auto & d : m_dependencies) {
            if (d->apply_callback(shared_from_this())) {
                // TODO: throw an exception if zero dims can't be propagated
            }
        }
    }

    void clean_dim_values() {
        for (auto & item : *this) {
            item.clear();
        }
    }
private:
    bool m_is_shape_like{false};

    std::map<Mask *, std::function<bool(Mask::Ptr)>> m_callbacks;

    std::vector<Mask *> m_dependencies;

    bool m_need_initialization{true};
};

std::ostream & operator<< (std::ostream & out, const Mask & mask);

extern template class VariantImpl<Mask::Ptr>;

template<>
class VariantWrapper<Mask::Ptr> : public VariantImpl<Mask::Ptr> {
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

Mask::Ptr getMask(const Output<const Node> & output);

Mask::Ptr getMask(const Output<Node> & output);

void setMask(Output<Node> output, const Mask::Ptr & mask);

}  // namespace ngraph
