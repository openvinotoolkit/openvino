// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief SpaceToDepth operation rearranges data from the spatial dimensions of the input tensor into depth dimension of the output tensor.
/// @details SpaceToDepth operation permutes element from the input tensor with shape [b, f, y, x]
/// to the output tensor where values from the input spatial dimensions y, x are moved to the new
/// depth dimension. Refer to the [ONNX* specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth)
/// for an example of the 4D input tensor case.
///
/// There are 2 attributes of this operation. The first attribute "block_size" specifies the size of the value block to be moved.
/// The depth dimension size must be evenly divided by (block_size ^ 2). This parameter is a positive integer with default value "1"
/// and no requiered. The second attribute "mode" specifies how the output depth dimension is gathered from block coordinates
/// and the old depth dimension. It's a string with non-default value. Required. Range of values:
///     - if mode is "blocks_first": the output depth is gathered from [block_size, block_size, f]
///     - if mode is "depth_first": the output depth is gathered from [f, block_size, block_size]
///
/// The operation is equivalent to the following transformation of the input tensor "data" with [y, x] spatial dimensions
/// of shape [b, f, y, x] to Z output tensor.
///
/// If "mode = blocks_first":
///
///     x' = reshape(data, [b, f, y/block_size, block_size, x/block_size, block_size])
///
///     x" = transpose(x',  [0, 3, 5, 1, 2, 4])
///
///     z = reshape(x", [b, f * (block_size ^ 2), y / block_size, x / block_size])
///
/// If "mode = depth_first":
///
///     x' = reshape(data, [b, f, D1/block_size, block_size, D2/block_size, block_size, ..., DK/block_size, block_size])
///
///     x" = transpose(x', [0, 1, 3, 5, 2, 4])
///
///     z = reshape(x", [b, f * (block_size ^ 2), y / block_size, x / block_size])

struct space_to_depth : public primitive_base<space_to_depth> {
    CLDNN_DECLARE_PRIMITIVE(space_to_depth)

    enum depth_mode {
        depth_first,
        blocks_first
    };

    /// @brief Constructs space_to_depth primitive.
    /// @param id This primitive id.
    /// @param input Input dictionary primitive id.
    /// @param depth_mode Depth mode (blocks_first / depth_first).
    /// @param block_size Block size (optional).
    space_to_depth(const primitive_id& id,
                   const primitive_id& input,
                   depth_mode mode,
                   const size_t block_size = 1,
                   const primitive_id& ext_prim_id = "",
                   const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding), mode(mode), block_size(block_size) {}

    /// @brief Depth mode.
    depth_mode mode;

    /// @brief Block size.
    size_t block_size;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
