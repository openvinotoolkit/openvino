// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief The BatchToSpace operation reshapes the "batch" dimension 0 into N - 1, N ∈ {4,5,6} dimensions of shape block_shape + [batch]
/// and interleaves these blocks back into the grid defined by the spatial dimensions [1, ..., N - 1], N ∈ {4,5,6}
/// to obtain a result with the same rank as data input.
/// The spatial dimensions of this intermediate result are then optionally cropped according to crops_begin and crops_end to produce the output.
/// @details The BatchToSpace operation is similar to the TensorFlow* operation
/// BatchToSpaceND (https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d)
/// There are 4 inputs of this operation:
/// 1) data - input N-D tensor [batch, D_1, D_2 ... D_{N-1}], N ∈ {4,5,6}. Required.
/// 2) block_shape - input 1-D tensor with shape [N], N ∈ {4,5,6}. Consists of block_sizes each of which specifies the size of the value block to be moved.
/// The batch dimension size must be evenly divided by (block_shape[1] * ... * block_shape[N - 1], N ∈ {4,5,6}).
/// All values must be >= 1 and required. block_shape[0] is expected to be 1.
/// 3) crops_begin - input 1-D tensor with shape [N], N ∈ {4,5,6}. Specifies amount to crop from the beginning along each axis of data input.
/// All values must be non-negative and required. crops_begin[0] is expected to be 0.
/// 4) crops_end - input 1-D tensor with shape [N], N ∈ {4,5,6}. Specifies the amount to crop from the ending along each axis of data input.
/// All values must be non-negative and required. crops_end[0] is expected to be 0.
/// 3-4 inputs required that crops_begin[i] + crops_end[i] < block_shape[i] * input_shape[i].
///
/// The operation is equivalent to the following transformation of the input tensors data with shape [batch, D_1, D_2 ... D_{N-1}], N ∈ {4,5,6}
/// and block_shape, crops_begin, crops_end of shape [N] to Y output tensor.
///
/// x' = reshape(`data`, [B_1, ..., B_{N - 1}, batch / (B_1 * ... B_{N - 1}), D_1, D_2, ..., D_{N - 1}]), where B_i = block_shape[i]
///
/// x'' = transpose(x', [N - 1, N, 0, N + 1, 1, ..., N + N - 2, N - 2])
///
/// x''' = reshape(x'', [batch / (B_1 * ... * B_{N - 1}), D_1 * B_1, D_2 * B_2, ... , D_{N - 1} * B_{N - 1}])
///
/// Crop the start and end of dimensions according to crops_begin, crops_end to produce the output of shape.
/// y = [batch / (B_1 * ... * B_{N - 1}), crop(D_1 * B_1, crops_begin[1], crops_end[1]), ... ,
///      crop(D_{N - 1} * B_{N - 1}, crops_begin[N - 1], crops_end[N - 1])]

struct batch_to_space : public primitive_base<batch_to_space> {
    CLDNN_DECLARE_PRIMITIVE(batch_to_space)

    batch_to_space() : primitive_base("", {}) {}

    /// @brief Constructs batch_to_space primitive.
    /// @param id This primitive id.
    /// @param input Input data primitive id.
    /// @param block_shape Array of block sizes
    /// @param crops_begin Amount to crop from the beginning along each axis of data input
    /// @param crops_end Amount to crop from the ending along each axis of data input
    batch_to_space(const primitive_id& id,
                   const input_info& input,
                   const tensor& block_shape,
                   const tensor& crops_begin,
                   const tensor& crops_end,
                   const tensor& out_size)
        : primitive_base(id, {input}),
          block_shape(block_shape),
          crops_begin(crops_begin),
          crops_end(crops_end),
          out_size(out_size),
          shape_constant(1) {}

    batch_to_space(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   const tensor& out_size)
        : primitive_base(id, inputs),
          block_shape(tensor()),
          crops_begin(tensor()),
          crops_end(tensor()),
          out_size(out_size),
          shape_constant(0) {}

    tensor block_shape;
    tensor crops_begin;
    tensor crops_end;
    tensor out_size;
    int64_t shape_constant;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, block_shape.hash());
        seed = hash_combine(seed, crops_begin.hash());
        seed = hash_combine(seed, crops_end.hash());
        seed = hash_combine(seed, shape_constant);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const batch_to_space>(rhs);

        return block_shape == rhs_casted.block_shape &&
               crops_begin == rhs_casted.crops_begin &&
               crops_end == rhs_casted.crops_end;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<batch_to_space>::save(ob);
        ob << block_shape;
        ob << crops_begin;
        ob << crops_end;
        ob << out_size;
        ob << shape_constant;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<batch_to_space>::load(ib);
        ib >> block_shape;
        ib >> crops_begin;
        ib >> crops_end;
        ib >> out_size;
        ib >> shape_constant;
    }
};
}  // namespace cldnn
