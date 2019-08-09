/*
// Copyright (c) 2019 Intel Corporation
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

#include "../C/reduce.h"
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Select mode for the @ref reduce layer
enum class reduce_mode : uint16_t {
    /// @brief Reduce max
    max = cldnn_reduce_max,
    /// @brief Reduce min
    min = cldnn_reduce_min,
    /// @brief Reduce mean
    mean = cldnn_reduce_mean,
    /// @brief Reduce prod
    prod = cldnn_reduce_prod,
    /// @brief Reduce sum
    sum = cldnn_reduce_sum,
    /// @brief Reduce and
    logical_and = cldnn_reduce_and,
    /// @brief Reduce or
    logical_or = cldnn_reduce_or,
    /// @brief Reduce  sum_square
    sum_square = cldnn_reduce_sum_square,
    /// @brief Reduce l1
    l1 = cldnn_reduce_l1,
    /// @brief Reduce l2
    l2 = cldnn_reduce_l2,
    /// @brief Reduce log_sum
    log_sum = cldnn_reduce_log_sum,
    /// @brief Reduce sum_exp
    log_sum_exp = cldnn_reduce_log_sum_exp
};

/// @brief Applies the specific reduction function along provided axes (second input) of the input tensor (first input).
/// @details
struct reduce : public primitive_base<reduce, CLDNN_PRIMITIVE_DESC(reduce)> {
    CLDNN_DECLARE_PRIMITIVE(reduce)

    enum reduce_axis {
        along_b = cldnn_reduce_along_b,
        along_f = cldnn_reduce_along_f,
        along_x = cldnn_reduce_along_x,
        along_y = cldnn_reduce_along_y,
        along_z = cldnn_reduce_along_z,
        along_w = cldnn_reduce_along_w
    };

    /// @brief Constructs reduce primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param keep_dims The axes which reduced
    reduce(const primitive_id& id, const primitive_id& input, const reduce_mode mode, const std::vector<uint16_t> axes,
           const int32_t keep_dims, const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding), mode(mode), axes(axes), keep_dims(keep_dims) {}

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{reduce}
    reduce(const dto* dto) : primitive_base(dto), mode(static_cast<reduce_mode>(dto->mode)),
                             axes(uint16_t_arr_to_vector(dto->axes)), keep_dims(dto->keep_dims) {}
    /// @brief Reduce operation type
    reduce_mode mode;
    /// @brief List of axes to reduce
    std::vector<uint16_t> axes;
    /// @brief Keep the reduced dimension or not, 1 mean keep reduced dimension
    int32_t keep_dims;

protected:
    void update_dto(dto& dto) const override {
        dto.mode = static_cast<int32_t>(mode);
        dto.keep_dims = keep_dims;
        dto.axes = uint16_t_vector_to_arr(axes);
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
