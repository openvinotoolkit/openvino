// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief SpaceToBatch operation divides "spatial" dimensions [1, ..., N - 1], N ∈ {4,5,6} of the data input
/// into a grid of blocks of shape block_shape, and interleaves these blocks with the batch dimension (0) such that in the output,
/// the spatial dimensions [1, ..., N - 1], N ∈ {4,5,6} correspond to the position within the grid,
/// and the batch dimension combines both the position within a spatial block and the original batch position.
/// Prior to division into blocks, the spatial dimensions of the input are optionally zero padded according to pads_begin and pads_end.
/// @details The SpaceToBatch operation is similar to the TensorFlow* operation SpaceToBatchND (https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd)
/// There are 4 inputs of this operation:
/// 1) data - input N-D tensor [batch, D_1, D_2 ... D_{N-1}], N ∈ {4,5,6}. Required.
/// 2) block_shape - input 1-D tensor with shape [N], N ∈ {4,5,6}. Consists of block_sizes each of which specifies the size of the value block to be moved.
/// All values must be >= 1 and required. block_shape[0] is expected to be 1.
/// 3) pads_begin - input 1-D tensor with shape [N], N ∈ {4,5,6}. Specifies the padding for the beginning along each axis of data input.
/// All values must be non-negative and required. pads_begin[0] is expected to be 0.
/// 4) pads_end - input 1-D tensor with shape [N], N ∈ {4,5,6}. Specifies the padding for the ending along each axis of data input.
/// All values must be non-negative and required. pads_end[0] is expected to be 0.
/// 3-4 inputs required that block_shape[i] divides data_shape[i] + pads_begin[i] + pads_end[i]
///
/// The operation is equivalent to the following transformation of the input tensor data of shape [batch, D_1, D_2 ... D_{N - 1}], N ∈ {4,5,6}
/// and block_shape, pads_begin, pads_end of shapes [N] to Y output tensor.
/// Zero-pad the start and end of dimensions [D_0, ..., D_{N - 1}] of the input according to `pads_begin` and `pads_end`
///
/// x' = reshape(x, [batch, (D_1 + P_1) / B_1, B_1, (D_2 + P_2) / B_2, B_2, ..., (D_{N - 1} + P_{N - 1}) / B_{N - 1}, B_{N - 1}]), where B_i = block_shape[i]
///
/// x'' = transpose(x',  [2, 4, ..., (N - 1) + (N - 1), 0, 1, 3, ..., N + (N - 1)])
///
/// y = reshape(x'', [batch * B_1 * ... * B_{N - 1}, (D_1 + P_1) / B_1, (D_2 + P_2) / B_2, ... , (D_{N - 1} + P_{N - 1}) / B_{N - 1}])

struct space_to_batch : public primitive_base<space_to_batch> {
    CLDNN_DECLARE_PRIMITIVE(space_to_batch)

    space_to_batch() : primitive_base("", {}) {}

    /// @brief Constructs space_to_batch primitive.
    /// @param id This primitive id.
    /// @param input Input data primitive id.
    /// @param block_shape Array of block sizes.
    /// @param pads_begin Amount to pad for the beginning along each axis of data input.
    /// @param pads_end Amount to pad for the ending along each axis of data input.
    /// @param out_size Size of output tensor.
    space_to_batch(const primitive_id& id,
                   const input_info& input,
                   const tensor& block_shape,
                   const tensor& pads_begin,
                   const tensor& pads_end,
                   const tensor& out_size)
        : primitive_base(id, {input}),
          block_shape(block_shape),
          pads_begin(pads_begin),
          pads_end(pads_end),
          out_size(out_size),
          shape_constant(1) {}

    space_to_batch(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   const tensor& out_size)
        : primitive_base(id, inputs),
          block_shape(tensor()),
          pads_begin(tensor()),
          pads_end(tensor()),
          out_size(out_size),
          shape_constant(0) {}

    tensor block_shape;
    tensor pads_begin;
    tensor pads_end;
    tensor out_size;
    int64_t shape_constant;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, block_shape.hash());
        seed = hash_combine(seed, pads_begin.hash());
        seed = hash_combine(seed, pads_end.hash());
        seed = hash_combine(seed, shape_constant);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const space_to_batch>(rhs);

        return block_shape == rhs_casted.block_shape &&
               pads_begin == rhs_casted.pads_begin &&
               pads_end == rhs_casted.pads_end &&
               shape_constant == rhs_casted.shape_constant;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<space_to_batch>::save(ob);
        ob << block_shape;
        ob << pads_begin;
        ob << pads_end;
        ob << out_size;
        ob << shape_constant;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<space_to_batch>::load(ib);
        ib >> block_shape;
        ib >> pads_begin;
        ib >> pads_end;
        ib >> out_size;
        ib >> shape_constant;
    }
};
}  // namespace cldnn
