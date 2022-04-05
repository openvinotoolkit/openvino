// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Select method for the @ref pooling layer.
enum class pooling_mode : int32_t {
    /// @brief Maximum-pooling method.
    max,
    /// @brief Average-pooling method - values.
    average,
    /// @brief Average-pooling method without values which are outside of the input.
    average_no_padding,
    /// @brief Maximum-pooling method with additional buffer to store argmax indices.
    max_with_argmax,
    /// @brief Pooling with bilinear interpolation.
    bilinear,
    /// @brief Deformable pooling with bilinear interpolation.
    deformable_bilinear
};

/// @brief Performs "pooling" operation which is a form of non-linear down-sampling.
/// @details Pools the input image by taking the max, average, etc. within regions.
struct pooling : public primitive_base<pooling> {
    CLDNN_DECLARE_PRIMITIVE(pooling)

    /// @brief Constructs pooling primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mode Pooling mode.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param size Pooling kernel size.
    /// @param pad Defines logical pad value added to input tensor.
    pooling(const primitive_id& id,
            const primitive_id& input,
            pooling_mode mode,
            const ov::Shape& size,
            const ov::Strides& stride,
            const ov::Shape& pad = {0, 0},
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          argmax(""),
          mode(static_cast<pooling_mode>(mode)),
          global_pooling(false),
          pad(pad),
          stride(stride),
          size(size),
          with_output_size(false),
          pad_end(size.size(), 0) {}

    /// @brief Constructs pooling primitive with argmax.
    /// @param id This primitive id.
    /// @param ext_prim_id
    /// @param input Input primitive id.
    /// @param argmax Primitive id which contains indices of each max pooling region.
    /// Indices must be in flattened bfyx format with no padding. Needs to be fp32 data type.
    /// @param mode Pooling mode.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param size Pooling kernel size.
    /// @param pad Defines logical pad value added to input tensor
    pooling(const primitive_id& id,
            const primitive_id& input,
            const primitive_id& argmax,
            pooling_mode mode,
            const ov::Shape& size,
            const ov::Strides& stride,
            const ov::Shape& pad = {0, 0},
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          argmax(argmax),
          mode(static_cast<pooling_mode>(mode)),
          global_pooling(false),
          pad(pad),
          stride(stride),
          size(size),
          with_output_size(false),
          pad_end(size.size(), 0) {}

    /// @brief Constructs pooling primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mode Pooling mode.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param size Pooling kernel size.
    /// @param pad Defines logical pad value added to input tensor.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    pooling(const primitive_id& id,
            const primitive_id& input,
            pooling_mode mode,
            const ov::Shape& size,
            const ov::Strides& stride,
            const ov::Shape& pad,
            tensor output_size,
            const data_types output_data_type,
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding, optional_data_type{output_data_type}),
          argmax(""),
          mode(static_cast<pooling_mode>(mode)),
          global_pooling(false),
          pad(pad),
          stride(stride),
          size(size),
          with_output_size(true),
          output_size(output_size),
          pad_end(size.size(), 0) {}

    /// @brief Constructs pooling primitive with argmax (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param argmax Primitive id which contains indices of each max pooling region.
    /// Indices must be in flattened bfyx format with no padding. Needs to be fp32 data type.
    /// @param mode Pooling mode.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param size Pooling kernel size.
    /// @param pad Defines logical pad value added to input tensor.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    pooling(const primitive_id& id,
            const primitive_id& input,
            const primitive_id& argmax,
            pooling_mode mode,
            const ov::Shape& size,
            const ov::Strides& stride,
            const ov::Shape& pad,
            tensor output_size,
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          argmax(argmax),
          mode(static_cast<pooling_mode>(mode)),
          global_pooling(false),
          pad(pad),
          stride(stride),
          size(size),
          with_output_size(true),
          output_size(output_size),
          pad_end(size.size(), 0) {}

    /// @brief Constructs pooling primitive with kernel size equal to the spatial dimension of input tensor.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param mode Pooling mode.
    pooling(const primitive_id& id,
            const primitive_id& input,
            pooling_mode mode,
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          argmax(""),
          mode(static_cast<pooling_mode>(mode)),
          global_pooling(true),
          pad({0, 0}),
          stride({1, 1}),
          size({0, 0}),
          with_output_size(false),
          pad_end(size.size(), 0) {}

    /// @brief Constructs pooling primitive that supports MaxPool features from opset8 (dilation and indices output).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param indices_output Indices output primitive id.
    /// @param size Pooling kernel size.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines index of next pixel to select when pooling.
    /// @param pad Defines logical pad value added to input tensor.
    /// @param pad_end Defines a shift, relative to the end of padding shape.
    /// @param axis First dimension of input that should be used to calculate the upper bound of index output.
    /// @param index_element_type Data type of index output.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    pooling(const primitive_id& id,
            const primitive_id& input,
            const primitive_id& indices_output,
            const ov::Shape& size,
            const ov::Strides& stride,
            const ov::Strides& dilation,
            const ov::Shape& pad,
            const ov::Shape& pad_end,
            int64_t axis,
            data_types index_element_type,
            tensor output_size,
            const data_types output_data_type,
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
            : primitive_base(id, {input, indices_output}, ext_prim_id, output_padding, optional_data_type{output_data_type}),
              argmax(""),
              indices_output(indices_output),
              mode(pooling_mode::max),
              global_pooling(false),
              pad(pad),
              stride(stride),
              dilation(dilation),
              size(size),
              with_output_size(true),
              output_size(output_size),
              pad_end(pad_end),
              axis(axis),
              index_element_type(index_element_type),
              maxPoolOpset8Features(true)
              {}

    /// @brief Primitive id which contains indices of each max pooling region.
    /// Indices must be in flattened bfyx format with no padding. Needs to be fp32 data type.
    primitive_id argmax;
    /// @brief Primitive id which contains indices output.
    primitive_id indices_output;
    /// @brief Pooling mode.
    pooling_mode mode;
    /// @brief Global pooling (kernel size is equal to the spatial dimension of input tensor)
    bool global_pooling;
    /// @brief Defines logical pad value added to input tensor.
    ov::Shape pad;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    ov::Strides stride;
    /// @brief Defines index of next pixel to select when pooling
    ov::Strides dilation;
    /// @brief Pooling kernel size.
    ov::Shape size;
    /// @brief Indicates that the primitive has user-defined output size (non-zero value).
    bool with_output_size;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief Defines a shift, relative to the end of padding shape.
    ov::Shape pad_end;
    /// @brief first dimension of input that should be used to calculate the upper bound of index output
    int64_t axis = 0;
    /// @brief type of index output
    data_types index_element_type = data_types::i32;
    bool maxPoolOpset8Features{false};

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!argmax.empty())
            ret.push_back(argmax);
        if (!indices_output.empty())
            ret.push_back(indices_output);
        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
