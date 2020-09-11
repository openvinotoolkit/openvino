/*
// Copyright (c) 2020 Intel Corporation
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
    /// @param input Input primitive id.
    /// @param input sequence_indicators primitive id.
    /// @param ctc_merge_repeated int
    ctc_greedy_decoder(const primitive_id& id,
        const primitive_id& input,
        const primitive_id& sequence_indicators,
        const bool ctc_merge_repeated,
        const data_types data_type,
        const tensor output_tensor,
        const padding& output_padding = padding())
        : primitive_base(id, { input, sequence_indicators },
            output_padding, optional_data_type{ data_type }),
        ctc_merge_repeated(ctc_merge_repeated),
        output_tensor(output_tensor)
    {}

    bool ctc_merge_repeated;
    tensor output_tensor;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
