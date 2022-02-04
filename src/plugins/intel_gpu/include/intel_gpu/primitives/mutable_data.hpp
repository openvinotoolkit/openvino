// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Provides mutable data.
/// @details This primitive allows to pass data which can be written to during training.
/// For example, weights and biases for scoring networks.
/// This primitive can be also set as other primitive's output. In this case the underlying buffer will be the same in mutable_data and preceding primitive.
struct mutable_data : public primitive_base<mutable_data> {
    CLDNN_DECLARE_PRIMITIVE(mutable_data)

    /// @brief Enum type to specify function for data filling.
    enum filler_type { no_fill, zero, one, xavier };

    /// @brief Constructs mutable_data primitive.
    /// @param id This primitive id.
    /// @param mem @ref memory object which contains data.
    /// @param filler_type @ref data filling function, default is zero
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    mutable_data(const primitive_id& id,
                 memory::ptr mem,
                 const primitive_id& ext_prim_id = "",
                 filler_type fill_type = filler_type::no_fill)
        : primitive_base(id, {}, ext_prim_id, padding()), mem(mem), fill_type(fill_type) {}

    /// @brief Constructs mutable_data primitive with inputs.
    /// @param id This primitive id.
    /// @param input Vector of input primitives ids.
    /// @param mem @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    /// @param filler_type @ref data filling function, default is zero
    mutable_data(const primitive_id& id,
                 const std::vector<primitive_id>& input,
                 memory::ptr mem,
                 const primitive_id& ext_prim_id = "",
                 filler_type fill_type = filler_type::no_fill)
        : primitive_base(id, {input}, ext_prim_id, padding()), mem(mem), fill_type(fill_type) {}

    /// @brief @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    memory::ptr mem;

    /// @brief Specifies function which will be used to fill weights.
    filler_type fill_type;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
