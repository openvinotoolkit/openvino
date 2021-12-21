// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

/// @brief Performs "max_unpooling" operation.
/// @details Reverse operation of max pooling, based on the argmax data where indices of each max pooling region are stored.
struct max_unpooling : public primitive_base<max_unpooling> {
    CLDNN_DECLARE_PRIMITIVE(max_unpooling)

    /// @brief Constructs max_unpooling primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param argmax Primitive id which contains indices of each max pooling region.
    /// Indices must be in flattened bfyx format with no padding. Needs to be fp32 data type.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// Used only for output size computation.
    /// @param size Pooling kernel size. Used only for output size computation.
    /// @param pad Defines logical pad value added to input tensor. Used only for output size computation.
    max_unpooling(const primitive_id& id,
                  const primitive_id& input,
                  const primitive_id& argmax,
                  const tensor& size,
                  const tensor& stride,
                  const tensor& pad = {0, 0, 0, 0},
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          argmax(argmax),
          pad(pad),
          stride(stride),
          size(size),
          with_output_size(false) {}

    /// @brief Constructs max_unpooling primitive (with provided output size)
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param argmax Primitive id which contains indices of each max pooling region.
    /// Indices must be in flattened bfyx format with no padding. Needs to be fp32 data type.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    max_unpooling(const primitive_id& id,
                  const primitive_id& input,
                  const primitive_id& argmax,
                  tensor output_size,
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          argmax(argmax),
          with_output_size(true),
          output_size(output_size) {}

    /// @brief Primitive id which contains indices of each max pooling region.
    /// Indices must be in flattened bfyx format with no padding. Needs to be fp32 data type.
    primitive_id argmax;
    /// @brief Defines logical pad value added to input tensor.
    tensor pad;
    /// @brief Defines shift in input buffer between adjacent calculations of output values. Used only for output size computation.
    tensor stride;
    /// @brief Pooling kernel size. Used only for output size computation.
    tensor size;
    /// @brief Indicates that the primitive has user-defined output size (non-zero value). Used only for output size computation.
    bool with_output_size;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return {argmax}; }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
