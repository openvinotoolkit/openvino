// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
                       const primitive_id& ext_prim_id = "",
                       const padding& output_padding = padding())
        : primitive_base(id, input, ext_prim_id, output_padding)
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
