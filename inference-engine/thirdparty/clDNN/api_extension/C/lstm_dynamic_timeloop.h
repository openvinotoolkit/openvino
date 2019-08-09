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
#include "api/C/cldnn.h"
/// @addtogroup c_api C API
/// @{
/// @addtogroup c_topology Network Topology
/// @{
/// @addtogroup c_primitives Primitives
/// @{

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Performs forward calcaulations of input gates for dynamic lstm layer.
/// @details The current implementation of LSTM_DYNAMIC is described the following equations.
///   it = f(Xt*(Wi^T) + Ht-1*Ri + Wbi)
///   ft = f(Xt*(Wf^T) + Ht-1*Rf + Wbf)
///   ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc)
///   Ct = ft (.) Ct-1 + it (.) ct
///   ot = f(Xt*(Wo^T) + Ht-1*Ro + Wbo)
///   Ht = ot (.) h(Ct)
/// Where f = Sigmoid, g = Tanh, and h = Tanh.
CLDNN_BEGIN_PRIMITIVE_DESC(lstm_dynamic_timeloop)

/// @brief Array of primitive ids containing recurrent weight matrices for input, output, forget, and cell gates.
cldnn_primitive_id recurrent;
/// @brief Primitive Id of mutable data primitive pointing to buffer, which will be filled with last hidden state.
cldnn_primitive_id last_hidden_state;
/// @brief Primitive Id of mutable data primitive pointing to buffer, which will be filled with last cell state.
cldnn_primitive_id last_cell_state;
/// @brief Array of primitive ids containing the initial value of the hidden data (Ht-1).
cldnn_primitive_id initial_hidden;
/// @brief Array of primitive ids containing the initial value of the cell state data (Ct-1).
cldnn_primitive_id initial_cell;
/// @brief Primitive id containing the dynamic sequence lengths.
cldnn_primitive_id dyn_length;
/// @brief Cell clip threshold T. It is applied to the input of activations [-T, T]. No clip is applied if it is not specified.
float clip;
/// @brief Couple the input and forget gates if input_forget is 1. Default is 0.
bool input_forget;
CLDNN_END_PRIMITIVE_DESC(lstm_dynamic_timeloop)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(lstm_dynamic_timeloop);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
