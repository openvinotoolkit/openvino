// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
/// @par Where:
struct region_yolo : public primitive_base<region_yolo> {
    CLDNN_DECLARE_PRIMITIVE(region_yolo)

    region_yolo() : primitive_base("", {}) {}

    /// @brief Constructs region_yolo primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param dimension Defines a scope of normalization (see #dimension).
    region_yolo(const primitive_id& id,
                const input_info& input,
                const uint32_t coords,
                const uint32_t classes,
                const uint32_t num,
                const std::vector<int64_t>& mask,
                const uint32_t mask_size,
                const int32_t axis,
                const int32_t end_axis,
                const bool do_softmax = true)
        : primitive_base(id, {input}),
          coords(coords),
          classes(classes),
          num(num),
          mask(mask),
          mask_size(mask_size),
          axis(axis),
          end_axis(end_axis),
          do_softmax(do_softmax) {}

    /// @brief Defines a scope of a region yolo normalization
    /// @details
    /// Specific behaviour is determined by these parameters, as follows:
    uint32_t coords = 0;
    uint32_t classes = 0;
    uint32_t num = 0;
    std::vector<int64_t> mask;
    uint32_t mask_size = 0;
    int32_t axis = 0;
    int32_t end_axis = 0;
    bool do_softmax = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, coords);
        seed = hash_combine(seed, classes);
        seed = hash_combine(seed, num);
        seed = hash_range(seed, mask.begin(), mask.end());
        seed = hash_combine(seed, mask_size);
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, end_axis);
        seed = hash_combine(seed, do_softmax);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const region_yolo>(rhs);

        return coords == rhs_casted.coords &&
               classes == rhs_casted.classes &&
               num == rhs_casted.num &&
               mask == rhs_casted.mask &&
               mask_size == rhs_casted.mask_size &&
               axis == rhs_casted.axis &&
               end_axis == rhs_casted.end_axis &&
               do_softmax == rhs_casted.do_softmax;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<region_yolo>::save(ob);
        ob << coords;
        ob << classes;
        ob << num;
        ob << mask;
        ob << mask_size;
        ob << axis;
        ob << end_axis;
        ob << do_softmax;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<region_yolo>::load(ib);
        ib >> coords;
        ib >> classes;
        ib >> num;
        ib >> mask;
        ib >> mask_size;
        ib >> axis;
        ib >> end_axis;
        ib >> do_softmax;
    }
};
}  // namespace cldnn
#pragma once
