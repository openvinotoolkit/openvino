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
#include "api/primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs forward calcaulations of input gates for dynamic lstm layer.
/// @details The current implementation of LSTM_DYNAMIC is described the following equations.
///   it = f(Xt*(Wi^T) + Ht-1*Ri + Wbi)
///   ft = f(Xt*(Wf^T) + Ht-1*Rf + Wbf)
///   ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc)
///   Ct = ft (.) Ct-1 + it (.) ct
///   ot = f(Xt*(Wo^T) + Ht-1*Ro + Wbo)
///   Ht = ot (.) h(Ct)
/// Where f = Sigmoid, g = Tanh, and h = Tanh.
struct lstm_dynamic_input : public primitive_base<lstm_dynamic_input> {
    CLDNN_DECLARE_PRIMITIVE(lstm_dynamic_input)

    /// @brief Constructs lstm_dynamic layer.
    /// @param id This primitive id.
    /// @param input Primitive id of input layer.
    /// @param dyn_length Primitive id of ilayer containg dynamic length values (shape: 1D).
    /// @param weights Primitive id containing weights data.
    /// @param recurrent Primitive id containing recurrent data.
    /// @param bias Primitive id containing bias data. Provide empty string if using lstm_dynamic without bias.
    lstm_dynamic_input(const primitive_id& id,
                       const primitive_id& input,
                       const primitive_id& dyn_length,
                       const primitive_id& weights,
                       const primitive_id& bias = "",
                       const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding), dyn_length(dyn_length), weights(weights), bias(bias) {}

    /// @brief Primitive id containing the dynamic sequence lengths.
    primitive_id dyn_length;
    /// @brief Primitive id containing weights data.
    primitive_id weights;
    /// @brief Primitive id containing bias data.
    primitive_id bias;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.push_back(dyn_length);
        ret.push_back(weights);

        if (!bias.empty()) {
            ret.push_back(bias);
        }
        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
