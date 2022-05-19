// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/shape.hpp>

#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Kind of DFT operation.
enum class dft_kind {
    forward,
    inverse,
};

/// @brief DFT primitive.
struct dft : public primitive_base<dft> {
    CLDNN_DECLARE_PRIMITIVE(dft)

    /// @brief Constructs DFT primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axes Axes to perform DFT.
    /// @param output_shape Output shape.
    /// @param kind Kind of DFT operation.
    dft(const primitive_id& id,
        const primitive_id& input,
        std::vector<int64_t>&& axes,
        const ov::Shape& output_shape,
        dft_kind kind,
        const primitive_id& ext_prim_id = {},
        const padding& output_padding = {})
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          axes(std::move(axes)),
          output_shape(output_shape),
          kind(kind) {}

    std::vector<int64_t> axes;
    ov::Shape output_shape;
    dft_kind kind;
};

}  // namespace cldnn
