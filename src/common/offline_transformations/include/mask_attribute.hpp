// Copyright (C) 2018-2025 Intel Corporation
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
#include <set>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/util/log.hpp"

namespace ov {

/**
 * @ingroup ov_runtime_attr_api
 * @brief each element in vector represents dimension and each element
 * in set is an id of dimensions which contains zeros.
 */
class Mask : public std::vector<std::set<uint64_t>>, public std::enable_shared_from_this<Mask> {
public:
    static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static const ::ov::DiscreteTypeInfo type_info_static{"Mask", "0"};
        return type_info_static;
    }

    using Ptr = std::shared_ptr<Mask>;

    Mask() = default;

    explicit Mask(const ov::PartialShape& shape) : std::vector<value_type>(shape.rank().get_length()) {}

    explicit Mask(const size_t& size) : std::vector<value_type>(size) {}

    explicit Mask(const size_t& size, const bool adjust_value)
        : std::vector<value_type>(size),
          m_adjust_value(adjust_value) {}

    explicit Mask(const std::vector<value_type> val) : std::vector<value_type>(val) {}

    Mask(std::initializer_list<std::initializer_list<uint64_t>> list) : std::vector<value_type>() {
        for (const auto& dim_values : list) {
            push_back(dim_values);
        }
    }

    bool all_dims_are_empty() const {
        return std::all_of(begin(), end(), [](const value_type& value) {
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

    bool is_shape_like() const {
        return m_is_shape_like;
    }

    void set_shape_like(bool flag) {
        m_is_shape_like = flag;
    }

    void copy_value_from_mask(Mask* const mask) {
        auto cur_mask_iter = begin();
        auto mask_iter = mask->begin();
        while (cur_mask_iter != end() && mask_iter != mask->end()) {
            *cur_mask_iter = *mask_iter;

            cur_mask_iter++;
            mask_iter++;
        }
    }

    /* Copy values from given mask in reversed order.
        param: mask - given mask.
    */
    void copy_value_from_mask_reversed(Mask* const mask) {
        auto cur_mask_iter = rbegin();
        auto mask_iter = mask->rbegin();
        while (cur_mask_iter != rend() && mask_iter != mask->rend()) {
            *cur_mask_iter = *mask_iter;

            cur_mask_iter++;
            mask_iter++;
        }
    }

    /* Copy values from given mask in reversed order, ignoring dimensions
        specified in idx_mask. Ignored dimensions keep prevous values.
        Dimensions in idx_mask correspond to dimensions of current mask.
        idx_mask could be reversed by param invert_mask.
        param: mask - given mask.
        param: idx_mask - current mask dimensions indexes which will be skipped during copying.
        param invert_mask - do mask need to be inverted. Default value == false.
    */
    void copy_value_from_mask_reversed_masked(Mask* const mask,
                                              const std::set<int64_t>& idx_mask,
                                              const bool invert_mask = false) {
        auto cur_mask_iter = rbegin();
        auto mask_iter = mask->rbegin();
        while (cur_mask_iter != rend() && mask_iter != mask->rend()) {
            const auto idx = rend() - cur_mask_iter - 1;
            if ((idx_mask.find(idx) != idx_mask.end()) == invert_mask)
                *cur_mask_iter = *mask_iter;

            cur_mask_iter++;
            mask_iter++;
        }
    }

    /* Intersents current mask with given mask alligning
        dimension starting from the end.
        param: mask - given mask.
        returns: intersected masks alligned from the end.
    */
    Mask::Ptr intersect_masks_reversed(Mask* const mask) const {
        auto result_mask = std::make_shared<Mask>(std::max(size(), mask->size()));
        auto result_iter = result_mask->rbegin();
        auto mask_1_iter = rbegin();
        auto mask_2_iter = mask->rbegin();

        while (mask_1_iter != rend() && mask_2_iter != mask->rend() && result_iter != result_mask->rend()) {
            // Merge mask dimension values for both masks
            // Example: (MaskValue[1,2,3,4], MaskValue[2,3]) -> MaskValue[2,3]
            for (const auto& value : *mask_1_iter) {
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

    /* Unions current mask with given mask alligning
        dimension starting from the end.
        param: mask - given mask.
        returns: united masks alligned from the end.
    */
    Mask::Ptr union_masks_reversed(Mask* const mask) const {
        auto result_mask = std::make_shared<Mask>(std::max(size(), mask->size()));
        auto result_iter = result_mask->rbegin();
        auto mask_1_iter = rbegin();
        auto mask_2_iter = mask->rbegin();

        while (mask_1_iter != rend() && mask_2_iter != mask->rend() && result_iter != result_mask->rend()) {
            // Union mask dimension values for both masks
            // Example: (MaskValue[1,2,3,4], MaskValue[2, 5]) -> MaskValue[1, 2, 3, 4, 5]
            for (const auto& value : *mask_1_iter) {
                result_iter->insert(value);
            }
            for (const auto& value : *mask_2_iter) {
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

    /*
       ov::Model copies values from mask,
       except mask[axis], where it selects values from mask[axis] set
       that are within [split_start, split_end) range
       param: mask - input mask.
       param: axis - axis, where the split happens
       param: split_start
       param: split_end
    */
    void copy_and_slice_mask_from(const Mask* const mask, int64_t axis, uint64_t split_start, uint64_t split_end) {
        if (size() < mask->size())
            resize(mask->size());
        for (size_t i = 0; i < size(); i++) {
            if (static_cast<int64_t>(i) == axis) {
                std::set<uint64_t> dst_set;
                const auto& src_set = mask->at(i);
                auto it = src_set.lower_bound(split_start);
                while (it != src_set.end() && *it < split_end)
                    dst_set.insert(*it++ - split_start);
                at(i) = dst_set;
            } else {
                at(i) = mask->at(i);
            }
        }
    }

    bool add_callback(const std::function<bool(Mask::Ptr)>& receive_callback, Mask::Ptr mask) {
#ifdef ENABLE_OPENVINO_DEBUG
        if (m_callbacks.find(mask.get()) != m_callbacks.end())
            OPENVINO_DEBUG("Attempt to rewrite callback, could lead to unexpected behaviour");
#endif
        m_callbacks[mask.get()] = receive_callback;
        m_dependencies.push_back(mask.get());
        return true;
    }

    /* Modify state of this mask by corresponding callback,
    which returns modifying success status (bool) and then
    modify all dependent masks by their corresponding callbacks*/
    bool apply_callback(Mask::Ptr mask) {
        // TODO: in case if callback returns false we need to propagate original value
        const auto& ref_state = Mask(*this);
        // Modify this mask by recived mask
        if (!m_callbacks.at(mask.get())(shared_from_this())) {
            return false;
        }
        // In case this mask already visited and didn't change by
        // callback call - stop recursion
        if (!m_need_initialization && *this == ref_state) {
            return true;
        }
        // Mark mask as visited
        m_need_initialization = false;
        // recursively apply callbacks for each dependent mask
        for (const auto& m_dependency : m_dependencies) {
            if (m_dependency == mask.get())
                continue;
            if (!m_dependency->apply_callback(shared_from_this())) {
                return false;
            }
        }

        return mask->apply_callback(shared_from_this());
    }

    void invalidate() {
        clean_dim_values();
        for (const auto& d : m_dependencies) {
            if (d->apply_callback(shared_from_this())) {
                // TODO: throw an exception if zero dims can't be propagated
            }
        }
    }

    void clean_dim_values() {
        for (auto& item : *this) {
            item.clear();
        }
    }

    /* Ask mask to update ther dependencies
    even if mask value wasn't changed on callback*/
    void initialize_dependencies() {
        m_need_initialization = true;
    }

    bool adjust_value() const {
        return m_adjust_value;
    }

private:
    bool m_is_shape_like{false};
    // Flag is true if this mask should be interpretated in special way:
    // Each value of the constant at index i decreasing by a number
    // of elements in i'th mask dimension during weights shrinking pass.
    // Only a 1D constants could be pruned in this way and
    // the number of mask dimensions should be equal to the number of elements
    // in the constant. The constant is typically interpetated as a shape
    // of some operation.
    bool m_adjust_value{false};

    // Masks dependent on this mask vs methods, specifying how
    // this mask will be modifed by correspondent dependent mask
    std::map<Mask*, std::function<bool(Mask::Ptr)>> m_callbacks;
    // Vector of all dependent masks
    std::vector<Mask*> m_dependencies;
    // Param used like visiting label (visited or not) during mask applying call
    bool m_need_initialization{true};
};

std::ostream& operator<<(std::ostream& out, const Mask& mask);

Mask::Ptr getMask(const Output<const Node>& output);

Mask::Ptr getMask(const Output<Node>& output);

void setMask(Output<Node> output, const Mask::Ptr& mask);

void setMask(Input<Node> node, const Mask::Ptr& mask);

#ifdef ENABLE_OPENVINO_DEBUG
/* Get mask which was defined on InitMasks matcher pass*/
Mask::Ptr getInitMask(const Output<Node>& output);

/* Set mask which was defined on InitMasks matcher pass*/
void setInitMask(Output<Node> output, const Mask::Ptr& mask);
#endif

}  // namespace ov
