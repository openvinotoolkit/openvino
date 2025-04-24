// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/op/space_to_depth.hpp"
#include "primitive.hpp"

namespace cldnn {
using SpaceToDepth = ov::op::v0::SpaceToDepth;

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

    space_to_depth() : primitive_base("", {}) {}

    /// @brief Constructs space_to_depth primitive.
    /// @param id This primitive id.
    /// @param input Input dictionary primitive id.
    /// @param depth_mode Depth mode (BLOCKS_FIRST / DEPTH_FIRST).
    /// @param block_size Block size (optional).
    space_to_depth(const primitive_id& id,
                   const input_info& input,
                   SpaceToDepth::SpaceToDepthMode mode,
                   const size_t block_size = 1)
        : primitive_base(id, {input}), mode(mode), block_size(block_size) {}

    /// @brief Depth mode.
    SpaceToDepth::SpaceToDepthMode mode = SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;

    /// @brief Block size.
    size_t block_size = 1;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, mode);
        seed = hash_combine(seed, block_size);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const space_to_depth>(rhs);

        return mode == rhs_casted.mode &&
               block_size == rhs_casted.block_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<space_to_depth>::save(ob);
        ob << make_data(&mode, sizeof(SpaceToDepth::SpaceToDepthMode));
        ob << block_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<space_to_depth>::load(ib);
        ib >> make_data(&mode, sizeof(SpaceToDepth::SpaceToDepthMode));
        ib >> block_size;
    }
};
}  // namespace cldnn
