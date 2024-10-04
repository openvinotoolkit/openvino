// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Computes sums of "bags" of embeddings, without instantiating the intermediate embeddings.
/// @details For each index in `indices` this operator gets values from `data` embedding table and sums all values belonging to each bag.
struct embedding_bag : public primitive_base<embedding_bag> {
    CLDNN_DECLARE_PRIMITIVE(embedding_bag)

    embedding_bag() : primitive_base("", {}) {}

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
                  const std::vector<input_info>& inputs,
                  const embedding_bag_type& type,
                  const tensor& output_shape,
                  const int32_t default_index = -1)
        : primitive_base(id, inputs), type(type), output_shape(output_shape), default_index(default_index) {}

    /// @brief Type of EmbeddingBag operation
    embedding_bag_type type;
    /// @brief Shape of output layout
    tensor output_shape;
    /// @brief Default index
    int32_t default_index = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, type);
        seed = hash_combine(seed, default_index);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const embedding_bag>(rhs);

        return type == rhs_casted.type &&
               default_index == rhs_casted.default_index;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<embedding_bag>::save(ob);
        ob << make_data(&type, sizeof(embedding_bag_type));
        ob << output_shape;
        ob << default_index;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<embedding_bag>::load(ib);
        ib >> make_data(&type, sizeof(embedding_bag_type));
        ib >> output_shape;
        ib >> default_index;
    }
};
}  // namespace cldnn
