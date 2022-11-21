// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief CTCLoss-4 primitive.
struct ctc_loss : primitive_base<ctc_loss> {
    CLDNN_DECLARE_PRIMITIVE(ctc_loss)

    /// @brief Constructs ctc_loss primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param preprocess_collapse_repeated Flag for preprocessing labels before loss calculation.
    /// @param ctc_merge_repeated Flag for merging repeated characters in a potential alignment.
    /// @param unique Flag to find unique elements in a target.
    ctc_loss(const primitive_id& id,
             const std::vector<input_info>& inputs,
             bool preprocess_collapse_repeated,
             bool ctc_merge_repeated,
             bool unique,
             const padding& output_padding = {})
        : primitive_base(id, inputs, {output_padding}),
          preprocess_collapse_repeated(preprocess_collapse_repeated),
          ctc_merge_repeated(ctc_merge_repeated),
          unique(unique) {}

    bool preprocess_collapse_repeated;
    bool ctc_merge_repeated;
    bool unique;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
