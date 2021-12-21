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
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, inputs, ext_prim_id, output_padding), type(type), output_shape(output_shape), default_index(default_index) {}

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
