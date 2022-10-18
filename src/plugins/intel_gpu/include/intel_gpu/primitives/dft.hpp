// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/shape.hpp>
#include <utility>

#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

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
        const primitive_id& input,
        std::vector<int64_t> axes,
        std::vector<int64_t> signal_size,
        const ov::Shape& output_shape,
        dft_direction direction,
        dft_mode mode,
        const padding& output_padding = {})
        : primitive_base(id, {input}, output_padding),
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
};

}  // namespace cldnn
