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

    std::vector<size_t> get_not_empty_dims() {
        std::vector<size_t> not_empty_dims;
        for (size_t i = 0; i < this->size(); i++) {
            if (!this->at(i).empty())
                not_empty_dims.push_back(i);
        }
        return not_empty_dims;
    }

    bool is_shape_like() const { return m_is_shape_like; }

    void set_shape_like(bool flag) { m_is_shape_like = flag; }

    void copy_value_from_mask(Mask *const mask) {
        auto cur_mask_iter = begin();
        auto mask_iter = mask->begin();
        while (cur_mask_iter != end() && mask_iter != mask->end()) {
            *cur_mask_iter = *mask_iter;

            cur_mask_iter++;
            mask_iter++;
        }
    }

    void copy_value_from_mask_reversed(Mask *const mask) {
        auto cur_mask_iter = rbegin();
        auto mask_iter = mask->rbegin();
        while (cur_mask_iter != rend() && mask_iter != mask->rend()) {
            *cur_mask_iter = *mask_iter;

            cur_mask_iter++;
            mask_iter++;
        }
    }

    Mask::Ptr intersect_masks_reversed(Mask *const mask) {
        auto result_mask = std::make_shared<Mask>(std::max(size(), mask->size()));
        auto result_iter = result_mask->rbegin();
        auto mask_1_iter = rbegin();
        auto mask_2_iter = mask->rbegin();

        while (mask_1_iter != rend() &&
               mask_2_iter != mask->rend()) {
            // Merge mask dimension values for both masks
            // Example: (MaskValue[1,2,3,4], MaskValue[2,3]) -> MaskValue[2,3]
            for (const auto & value : *mask_1_iter) {
                if (mask_2_iter->count(value)) {
                    result_iter->insert(value);
                }
            }

            result_iter++;
            mask_1_iter++;
            mask_2_iter++;
        }
        return result_mask;
    }

    Mask::Ptr union_masks_reversed(Mask *const mask) {
        auto result_mask = std::make_shared<Mask>(std::max(size(), mask->size()));
        auto result_iter = result_mask->rbegin();
        auto mask_1_iter = rbegin();
        auto mask_2_iter = mask->rbegin();

        while (mask_1_iter != rend() &&
               mask_2_iter != mask->rend()) {
            // Union mask dimension values for both masks
            // Example: (MaskValue[1,2,3,4], MaskValue[2, 5]) -> MaskValue[1, 2, 3, 4, 5]
            for (const auto & value : *mask_1_iter) {
                result_iter->insert(value);
            }
            for (const auto & value : *mask_2_iter) {
                if (!result_iter->count(value)) {
                    result_iter->insert(value);
                }
            }

            result_iter++;
            mask_1_iter++;
            mask_2_iter++;
        }
        return result_mask;
    }

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
