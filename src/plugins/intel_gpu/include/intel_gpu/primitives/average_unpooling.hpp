// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs "average_unpooling" operation.
/// @details Reverse operation of average pooling.
/// Each element in every pooling window is filled with output / window size value. In case of window overlap the elements are added.
struct average_unpooling : public primitive_base<average_unpooling> {
    CLDNN_DECLARE_PRIMITIVE(average_unpooling)

    /// @brief Constructs average_unpooling primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_size Size of input for average pooling forward.
    /// @param stride Defines shift in output buffer.
    /// @param size Pooling kernel size.
    average_unpooling(
        const primitive_id& id,
        const input_info& input,
        const tensor output_size,
        const tensor& size,
        const tensor& stride,
        const primitive_id& ext_prim_id = "",
        const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, {output_padding}), stride(stride), size(size), output_size(output_size) {}

    /// @brief Defines shift in output buffer.
    tensor stride;
    /// @brief Pooling kernel size.
    tensor size;
    /// @brief Output size of this primitive.
    tensor output_size;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
