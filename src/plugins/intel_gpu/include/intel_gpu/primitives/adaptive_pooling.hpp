// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

enum class adaptive_pooling_mode : int32_t {
    max,
    average
};

struct adaptive_pooling : public primitive_base<adaptive_pooling> {
    CLDNN_DECLARE_PRIMITIVE(adaptive_pooling)

    adaptive_pooling() : primitive_base("", {}),
                         mode{adaptive_pooling_mode::average},
                         output_size{} {}

    /// @brief Constructs AdaptiveAvgPooling primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_size Output data size of the primitive
    adaptive_pooling(const primitive_id &id,
                     const input_info &input,
                     tensor output_size)
            : primitive_base(id, {input}),
              mode{adaptive_pooling_mode::average},
              output_size{output_size} {}

    /// @brief Constructs AdaptiveMaxPooling primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_shape Output shape primitive id.
    /// @param output_size Output data size of the primitive
    /// @param indices_output Indices output primitive id.
    /// @param index_element_type Data type of indices output.
    adaptive_pooling(const primitive_id &id,
                     const input_info &input,
                     tensor output_size,
                     const primitive_id &indices_output,
                     data_types index_element_type)
            : primitive_base(id, {input, indices_output}),
              mode{adaptive_pooling_mode::max},
              output_size{output_size},
              indices_output{indices_output},
              index_element_type{index_element_type} {}

    /// @brief Constructs AdaptiveAvgPooling primitive for dynamic shape.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_shape Output shape (pooled vector) primitive id.
    adaptive_pooling(const primitive_id &id,
                     const input_info &input,
                     const input_info &output_shape)
            : primitive_base(id, {input, output_shape}),
              mode{adaptive_pooling_mode::average},
              output_size{tensor(0)} {}

    /// @brief Constructs AdaptiveMaxPooling primitive for dynamic shape.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_shape Output shape (pooled vector) primitive id.
    /// @param index_element_type Data type of indices output.
    adaptive_pooling(const primitive_id &id,
                     const input_info &input,
                     const input_info &output_shape,
                     data_types index_element_type,
                     data_types output_data_type = data_types::i32,
                     const size_t num_outputs = 1)
            : primitive_base(id, {input, output_shape}, num_outputs, {optional_data_type{output_data_type}}),
              mode{adaptive_pooling_mode::max},
              output_size{tensor(0)},
              indices_output{""},
              index_element_type{index_element_type} {}

    adaptive_pooling_mode mode;
    tensor output_size;
    primitive_id indices_output;
    data_types index_element_type{data_types::i64};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, mode);
        seed = hash_combine(seed, index_element_type);
        seed = hash_combine(seed, indices_output.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const adaptive_pooling>(rhs);

        return mode == rhs_casted.mode &&
               indices_output == rhs_casted.indices_output &&
               index_element_type == rhs_casted.index_element_type;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<adaptive_pooling>::save(ob);
        ob << make_data(&mode, sizeof(adaptive_pooling_mode));
        ob << output_size;
        ob << indices_output;
        ob << make_data(&index_element_type, sizeof(data_types));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<adaptive_pooling>::load(ib);
        ib >> make_data(&mode, sizeof(adaptive_pooling_mode));
        ib >> output_size;
        ib >> indices_output;
        ib >> make_data(&index_element_type, sizeof(data_types));
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        if (!indices_output.empty())
            ret.push_back(indices_output);
        return ret;
    }
};
}  // namespace cldnn
