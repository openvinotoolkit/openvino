// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct gather_nd : public primitive_base<gather_nd> {
    CLDNN_DECLARE_PRIMITIVE(gather_nd)

    gather_nd() : primitive_base("", {}) {}

    /// @brief Constructs gather_nd primitive.
    ///
    /// @param id                   This primitive id.
    /// @param data                 Input data primitive id.
    /// @param indices              Input indexes primitive id.
    /// @param input_rank           Rank of input data.
    /// @param indices_rank         Rank of indices.
    /// @param batch_dims           batch_dims as an attribute of GatherND. Optional.
    /// @param batch_merged_output  batched output shape is merged as a dimention for v5.
    ///                             In case of output{3, 2, 4, 5} at batch_dims = 2, real output shape should be {6, 4, 5}.
    ///                             This should be false for v8.
    ///                             For batch_dims < 2, This doesn't have any meaning.
    gather_nd(const primitive_id& id,
              const input_info& data,
              const input_info& indices,
              const uint8_t input_rank,
              const uint8_t indices_rank,
              const uint8_t batch_dims = 0,
              const bool batch_merged_output = true)
        : primitive_base(id, {data, indices}),
                         input_rank(input_rank),
                         indices_rank(indices_rank),
                         batch_dims(batch_dims),
                         batch_merged_output(batch_merged_output) {}

    /// @brief GatherND input_rank
    uint8_t input_rank;

    /// @brief GatherND indices_rank
    uint8_t indices_rank;

    /// @brief GatherND batch_dims
    uint8_t batch_dims;

    /// @brief GatherND batch_merged_output
    bool batch_merged_output = true;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, indices_rank);
        seed = hash_combine(seed, batch_dims);
        seed = hash_combine(seed, batch_merged_output);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const gather_nd>(rhs);

        return input_rank == rhs_casted.input_rank &&
               indices_rank == rhs_casted.indices_rank &&
               batch_dims == rhs_casted.batch_dims &&
               batch_merged_output == rhs_casted.batch_merged_output;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gather_nd>::save(ob);
        ob << input_rank;
        ob << indices_rank;
        ob << batch_dims;
        ob << batch_merged_output;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gather_nd>::load(ib);
        ib >> input_rank;
        ib >> indices_rank;
        ib >> batch_dims;
        ib >> batch_merged_output;
    }
};
}  // namespace cldnn
