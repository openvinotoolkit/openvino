/*
// Copyright (c) 2016 Intel Corporation
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
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{
/// @brief Type of gemm that will be added to the input by border layer / primitive.

/// @brief Adds gemm  input.
///
/// @details General Matrix Multiplication witch batch support,
///          A(B,Z,X)xA2(B,Y,Z)=C(B,X,Y)
/// @n
/// @n@b Requirements:
/// @n - @c input - first matrix
/// @n - @c input2 - second matrix
/// @n - @c optional: input3 matrix, alpha, beta, transpose
/// @n - @c computations with optional params: output = alpha x (input3 x beta + input x input2)
/// @n - @c transpose params tranposing second matrix <-TODO

struct gemm : public primitive_base<gemm> {
    CLDNN_DECLARE_PRIMITIVE(gemm)

    /// @brief Constructs gemm layer.
    /// @brief Primitive id containing first matrix
    /// @brief Primitive id containing second matrix
    /// @brief Flag for transposing first input matrix
    /// @brief Flag for transposing second input matrix
    /// @brief Variable containing ALPHA parameter
    /// @brief Variable containing BETA parameter

    gemm(const primitive_id& id,
         const std::vector<primitive_id>& inputs,
         const bool transpose_input0 = false,
         const bool transpose_input1 = false,
         const float alpha = 1.0f,
         const float beta = 0.0f,
         const padding& output_padding = padding())
        : primitive_base(id, inputs, output_padding),
          transpose_input0(transpose_input0),
          transpose_input1(transpose_input1),
          alpha(alpha),
          beta(beta) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            throw std::invalid_argument("Invalid inputs count - gemm expects either two or three inputs");
        }
    }

    /// @brief Flag for transposing first input matrix
    bool transpose_input0;
    /// @brief Flag for transposing second input matrix
    bool transpose_input1;
    /// @brief Variable containing ALPHA parameter
    float alpha;
    /// @brief Variable containing BETA parameter
    float beta;
};

}  // namespace cldnn

/// @}
/// @}
/// @}
