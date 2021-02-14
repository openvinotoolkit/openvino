/*
// Copyright (c) 2020-2021 Intel Corporation
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

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief CTC greedy decoder primitve
struct ctc_greedy_decoder : public primitive_base<ctc_greedy_decoder> {
    CLDNN_DECLARE_PRIMITIVE(ctc_greedy_decoder)

    /// @brief Constructs ctc_greedy_decoder primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id (input, sequence_indicators, second_output(optional)).
    /// @param blank_index Specifies the class index to use for the blank class.
    /// @param ctc_merge_repeated Flag for merging repeated labels during the CTC calculation
    ctc_greedy_decoder(const primitive_id& id,
                       const std::vector<primitive_id>& input,
                       const uint32_t blank_index,
                       const bool ctc_merge_repeated,
                       const tensor output_tensor,
                       const padding& output_padding = padding())
        : primitive_base(id, input, output_padding)
        , blank_index(blank_index)
        , ctc_merge_repeated(ctc_merge_repeated)
        , output_tensor(output_tensor) {}

    uint32_t blank_index;
    bool ctc_merge_repeated;
    tensor output_tensor;
    primitive_id second_output;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
