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

/// @brief Computes sums of "bags" of embeddings, without instantiating the intermediate embeddings.
/// @details For each index in `indices` this operator gets values from `data` embedding table and sums all values belonging to each bag.
struct embedding_bag : public primitive_base<embedding_bag> {
    CLDNN_DECLARE_PRIMITIVE(embedding_bag)

    /// @brief Select type of embedding_bag operation
    enum embedding_bag_type {
        packed_sum,
        offsets_sum,
        segments_sum
    };

    /// @brief Constructs embedding_bag primitive.
    /// @param id This primitive id.
    /// @param inputs Vector with different inputs.
    /// @param output_shape Tensor with shape of output layout
    /// @param default_index default index in embedding table to fill empty "bags"
    embedding_bag(const primitive_id& id,
                  const std::vector<primitive_id>& inputs,
                  const embedding_bag_type& type,
                  const tensor& output_shape,
                  const int32_t default_index = -1,
                  const padding& output_padding = padding())
        : primitive_base(id, inputs, output_padding), type(type), output_shape(output_shape), default_index(default_index) {}

    /// @brief Type of EmbeddingBag operation
    embedding_bag_type type;
    /// @brief Shape of output layout
    tensor output_shape;
    /// @brief Default index
    int32_t default_index;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
