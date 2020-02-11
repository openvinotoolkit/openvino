/*
// Copyright (c) 2016-2019 Intel Corporation
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

/// @brief Select mode for the @ref eltwise layer.
enum class eltwise_mode : int32_t {
    /// @brief Eltwise sum.
    sum,
    /// @brief Eltwise subtract.
    sub,
    /// @brief Eltwise max.
    max,
    /// @brief Eltwise product (Hadamard).
    prod,
    /// @brief Eltwise div.
    div,
    /// @brief Eltwise min.
    min,
    /// @brief Eltwise pow.
    pow,
    /// @brief Eltwise squared diff.
    squared_diff,
    /// @brief Eltwise mod.
    mod,
    /// @brief Eltwise equal.
    eq,
    /// @brief Eltwise not equal.
    ne,
    /// @brief Eltwise less.
    lt,
    /// @brief Eltwise less of equal.
    le,
    /// @brief Eltwise greater.
    gt,
    /// @brief Eltwise greater or equal.
    ge,
    /// @brief Eltwise and.
    logic_and,
    /// @brief Eltwise or.
    logic_or,
    /// @brief Eltwise XOR.
    logic_xor,
    /// @brief Eltwise floormod.
    floor_mod
};

/// @brief Performs elementwise operations (sum, subtract, max or product) on two input primitives
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @notes
/// - both inputs have to have equal sizes in all dimensions or the input tensors are broadcastable
///   to the same shape in which the size of each dimention is a max. of input sizes on this dimension)
/// - format of both inputs has to be the same
/// - when using integer types, only following eltwise modes are supported: sum, sub, prod, div
struct eltwise : public primitive_base<eltwise> {
    CLDNN_DECLARE_PRIMITIVE(eltwise)

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param input2 Second input primitive id with values needed for eltwise computation.
    /// @param mode Eltwise mode.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    eltwise(const primitive_id& id,
            const primitive_id& input,
            const primitive_id& input2,
            eltwise_mode mode,
            const padding& output_padding = padding())
        : primitive_base(id, {input, input2}, output_padding),
          output_calibration_factors(""),
          output_quantization_factor(1.0f),
          input_quantization_factors(0),
          mode(mode),
          coefficients(std::vector<float>(0)),
          stride(std::vector<tensor>(0)),
          inputs_calibration_factors(std::vector<primitive_id>(0)) {}

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param input2 Second input primitive id with values needed for eltwise computation.
    /// @param stride Defines shift in input buffers between adjacent calculations of output values.
    /// @param mode Eltwise mode.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    eltwise(const primitive_id& id,
            const primitive_id& input,
            const primitive_id& input2,
            std::vector<tensor> stride,
            eltwise_mode mode,
            const padding& output_padding = padding())
        : primitive_base(id, {input, input2}, output_padding),
          output_calibration_factors(""),
          output_quantization_factor(1.0f),
          input_quantization_factors(0),
          mode(mode),
          coefficients(std::vector<float>(0)),
          stride(stride),
          inputs_calibration_factors(std::vector<primitive_id>(0)) {}

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param mode Eltwise mode.
    /// @param data_type Expected output data type.
    eltwise(const primitive_id& id,
            const std::vector<primitive_id>& inputs,
            eltwise_mode mode,
            data_types data_type,
            const padding& output_padding = padding())
        : primitive_base(id, inputs, output_padding, optional_data_type{data_type}),
          output_calibration_factors(""),
          output_quantization_factor(1.0f),
          input_quantization_factors(0),
          mode(mode),
          coefficients(std::vector<float>(0)),
          stride(std::vector<tensor>(0)),
          inputs_calibration_factors(std::vector<primitive_id>(0)) {}

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param mode Eltwise mode.
    eltwise(const primitive_id& id,
            const std::vector<primitive_id>& inputs,
            eltwise_mode mode,
            const padding& output_padding = padding())
        : primitive_base(id, inputs, output_padding),
          output_calibration_factors(""),
          output_quantization_factor(1.0f),
          input_quantization_factors(0),
          mode(mode),
          coefficients(std::vector<float>(0)),
          stride(std::vector<tensor>(0)),
          inputs_calibration_factors(std::vector<primitive_id>(0)) {}

    /// @brief Constructs eltwise primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param coefficients Blob-wise coefficient for SUM operation
    /// @param mode Eltwise mode.
    eltwise(const primitive_id& id,
            const std::vector<primitive_id>& inputs,
            eltwise_mode mode,
            const std::vector<float>& coefficients,
            data_types data_type,
            const padding& output_padding = padding())
        : primitive_base(id, inputs, output_padding, optional_data_type{data_type}),
          output_calibration_factors(""),
          output_quantization_factor(1.0f),
          input_quantization_factors(0),
          mode(mode),
          coefficients(coefficients),
          stride(std::vector<tensor>(0)),
          inputs_calibration_factors(std::vector<primitive_id>(0)) {
        if (mode == eltwise_mode::sum && !coefficients.empty() && coefficients.size() != inputs.size()) {
            throw std::invalid_argument("Invalid eltwise sum coefficients count (should be equal to 0 or input.size)");
        }
        if (mode != eltwise_mode::sum && !coefficients.empty()) {
            throw std::invalid_argument("Only eltwise sum operation supports blob-wise coefficients");
        }
    }

    /// @brief Primitive id containing output quanitization factors per output feature map.
    primitive_id output_calibration_factors;
    /// @brief Output quantization factor
    float output_quantization_factor;
    /// @brief List of quantization factors per input.
    std::vector<float> input_quantization_factors;
    /// @param mode Eltwise mode.
    eltwise_mode mode;
    /// @param coefficients Blob-wise coefficient for SUM operation.
    std::vector<float> coefficients;
    /// @brief Defines shift in input buffers between adjacent calculations of output values.
    std::vector<tensor> stride;
    /// @brief List of primitive ids containing input quantization factors per feature map, one primitive id for each input.
    const primitive_id_arr inputs_calibration_factors;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!output_calibration_factors.empty())
            ret.push_back(output_calibration_factors);

        for (auto& icf : inputs_calibration_factors) ret.push_back(std::ref(icf));

        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
