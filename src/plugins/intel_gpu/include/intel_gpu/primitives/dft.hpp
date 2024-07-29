// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/shape.hpp>
#include <utility>

#include "primitive.hpp"

namespace cldnn {

/// @brief Direction of DFT operation.
enum class dft_direction {
    forward,
    inverse,
};

/// @brief Mode of DFT operation.
enum class dft_mode {
    complex,
    real,
};

/// @brief DFT primitive.
struct dft : public primitive_base<dft> {
    CLDNN_DECLARE_PRIMITIVE(dft)

    dft() : primitive_base("", {}) {}

    /// @brief Constructs DFT primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axes Axes to perform DFT.
    /// @param signal_size Signal sizes for 'axes'.
    /// @param output_shape Output shape.
    /// @param direction Direction of DFT operation.
    /// @param mode Mode of DFT operation.
    dft(const primitive_id& id,
        const input_info& input,
        std::vector<int64_t> axes,
        std::vector<int64_t> signal_size,
        const ov::Shape& output_shape,
        dft_direction direction,
        dft_mode mode)
        : primitive_base(id, {input}),
          axes(std::move(axes)),
          signal_size(std::move(signal_size)),
          output_shape(output_shape),
          direction(direction),
          mode(mode) {}

    /// @brief Constructs DFT primitive for dynamic shape input. # of input is 2.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axes Axes to perform DFT.
    /// @param direction Direction of DFT operation.
    /// @param mode Mode of DFT operation.
    dft(const primitive_id& id,
        const input_info& input,
        const input_info& axes,
        std::vector<int64_t> constant_axes,
        dft_direction direction,
        dft_mode mode)
        : primitive_base(id, {input, axes}),
          axes(constant_axes),
          signal_size({}),
          output_shape(ov::Shape(0)),
          direction(direction),
          mode(mode) {}

    /// @brief Constructs DFT primitive for dynamic shape input. # of input is 3.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axes Axes to perform DFT.
    /// @param signal_size Signal sizes for 'axes'.
    /// @param direction Direction of DFT operation.
    /// @param mode Mode of DFT operation.
    dft(const primitive_id& id,
        const input_info& input,
        const input_info& axes,
        const input_info& signal_size,
        std::vector<int64_t> constant_axes,
        std::vector<int64_t> constant_signal_size,
        dft_direction direction,
        dft_mode mode)
        : primitive_base(id, {input, axes, signal_size}),
          axes(constant_axes),
          signal_size(constant_signal_size),
          output_shape(ov::Shape(0)),
          direction(direction),
          mode(mode) {}

    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::Shape output_shape;
    dft_direction direction = dft_direction::forward;
    dft_mode mode = dft_mode::complex;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, axes.begin(), axes.end());
        seed = hash_range(seed, signal_size.begin(), signal_size.end());
        seed = hash_combine(seed, direction);
        seed = hash_combine(seed, mode);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const dft>(rhs);

        return axes == rhs_casted.axes &&
               signal_size == rhs_casted.signal_size &&
               direction == rhs_casted.direction &&
               mode == rhs_casted.mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<dft>::save(ob);
        ob << axes;
        ob << signal_size;
        ob << output_shape;
        ob << make_data(&direction, sizeof(dft_direction));
        ob << make_data(&mode, sizeof(dft_mode));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<dft>::load(ib);
        ib >> axes;
        ib >> signal_size;
        ib >> output_shape;
        ib >> make_data(&direction, sizeof(dft_direction));
        ib >> make_data(&mode, sizeof(dft_mode));
    }
};

}  // namespace cldnn
