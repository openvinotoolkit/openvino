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
        dft_mode mode,
        const padding& output_padding = {})
        : primitive_base(id, {input}, {output_padding}),
          axes(std::move(axes)),
          signal_size(std::move(signal_size)),
          output_shape(output_shape),
          direction(direction),
          mode(mode) {}

    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::Shape output_shape;
    dft_direction direction;
    dft_mode mode;

    size_t hash() const override {
        if (!seed) {
            seed = hash_range(seed, axes.begin(), axes.end());
            seed = hash_range(seed, signal_size.begin(), signal_size.end());
            seed = hash_combine(seed, direction);
            seed = hash_combine(seed, mode);
        }
        return seed;
    }
};

}  // namespace cldnn
