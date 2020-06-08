/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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

    /// @brief Constructs space_to_batch primitive.
    /// @param id This primitive id.
    /// @param input Input data primitive id.
    /// @param block_shape_id Array of block sizes primitive id
    /// @param pads_begin_id Amount to pad for the beginning along each axis of data input primitive id
    /// @param pads_end_id Amount to pad for the ending along each axis of data input primitive id
    space_to_batch(const primitive_id& id,
                   const primitive_id& input,
                   const primitive_id& block_shape_id,
                   const primitive_id& pads_begin_id,
                   const primitive_id& pads_end_id,
                   const padding& output_padding = padding())
        : primitive_base(id, {input, block_shape_id, pads_begin_id, pads_end_id}, output_padding){}
};
/// @}
/// @}
/// @}
}  // namespace cldnn
