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

    /// @brief Performs gather tree
    ///
    /// @details Performs gather tree
struct gather_tree : public primitive_base<gather_tree> {
    CLDNN_DECLARE_PRIMITIVE(gather_tree)

        /// @brief Constructs gather tree primitive / layer.
        ///
        /// @param id                      An identifier of new primitive.
        /// @param step_input              An identifier of primitive which is an step input
        /// @param parent_input            An identifier of primitive which is an parent input
        /// @param step_seq_len_input      An identifier of primitive which is an input that contains
        ///                                lengths of step sequence (per batch) to perform
        /// @param end_token               An identifier of primitive which is an input that contains
        ///                                a value of the end_token
        /// @param output_padding          Optional padding for output from primitive
        gather_tree(const primitive_id& id,
                    const input_info& step_input,
                    const input_info& parent_input,
                    const input_info& max_seq_len_input,
                    const input_info& end_token,
                    const primitive_id& ext_prim_id = "",
                    const padding& output_padding = padding())
            : primitive_base(id, { step_input, parent_input, max_seq_len_input, end_token }, ext_prim_id, {output_padding}) {}
};
    /// @}
    /// @}
    /// @}
}  // namespace cldnn
