// Copyright (C) 2022-2023 Intel Corporation
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

    adaptive_pooling_mode mode;
    tensor output_size;
    primitive_id indices_output;
    data_types index_element_type{data_types::i64};

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!indices_output.empty())
            ret.push_back(indices_output);
        return ret;
    }
};
}  // namespace cldnn
