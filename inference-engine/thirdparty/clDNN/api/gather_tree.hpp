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
            const primitive_id& step_input,
            const primitive_id& parent_input,
            const primitive_id& max_seq_len_input,
            const primitive_id& end_token,
            const padding& output_padding = padding())
        : primitive_base(id, { step_input, parent_input, max_seq_len_input, end_token }, output_padding) {}
};
    /// @}
    /// @}
    /// @}
}  // namespace cldnn
