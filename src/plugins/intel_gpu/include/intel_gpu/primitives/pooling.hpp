// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"

#include "openvino/op/util/attr_types.hpp"

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
            const ov::Shape& pads_begin = {0, 0},
            const ov::Shape& pads_end = {0, 0},
            ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT,
            ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
            const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          mode(static_cast<pooling_mode>(mode)),
          size(size),
          stride(stride),
          pads_begin(pads_begin),
          pads_end(pads_end),
          auto_pad(auto_pad),
          rounding_type(rounding_type),
          with_output_size(false) {}

    /// @brief Constructs pooling primitive with known output shape.
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
            const ov::Shape& pads_begin,
            const ov::Shape& pads_end,
            tensor output_size,
            const data_types output_data_type,
            const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding, optional_data_type{output_data_type}),
          mode(static_cast<pooling_mode>(mode)),
          size(size),
          stride(stride),
          pads_begin(pads_begin),
          pads_end(pads_end),
          auto_pad(ov::op::PadType::EXPLICIT),
          rounding_type(ov::op::RoundingType::CEIL),
          with_output_size(true),
          output_size(output_size) {}

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
            const ov::Shape& pads_begin,
            const ov::Shape& pads_end,
            ov::op::PadType auto_pad,
            ov::op::RoundingType rounding_type,
            int64_t axis,
            data_types index_element_type,
            tensor output_size,
            const data_types output_data_type,
            const padding& output_padding = padding())
            : primitive_base(id, {input, indices_output}, output_padding, optional_data_type{output_data_type}),
              indices_output(indices_output),
              mode(pooling_mode::max),
              size(size),
              stride(stride),
              dilation(dilation),
              pads_begin(pads_begin),
              pads_end(pads_end),
              auto_pad(auto_pad),
              rounding_type(rounding_type),
              axis(axis),
              with_output_size(true),
              output_size(output_size),
              index_element_type(index_element_type),
              maxPoolOpset8Features(true) {}

    /// @brief Primitive id which contains indices output.
    primitive_id indices_output;
    /// @brief Pooling mode.
    pooling_mode mode;
    /// @brief Pooling kernel size.
    ov::Shape size;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    ov::Strides stride;
    /// @brief Defines index of next pixel to select when pooling
    ov::Strides dilation;
    /// @brief Defines logical pad value added to input tensor.
    ov::Shape pads_begin;
    /// @brief Defines a shift, relative to the end of padding shape.
    ov::Shape pads_end;
    /// @brief Defines how the padding is calculated.
    ov::op::PadType auto_pad;
    /// @brief Defines a type of rounding to be applied.
    ov::op::RoundingType rounding_type;
    /// @brief first dimension of input that should be used to calculate the upper bound of index output.
    int64_t axis = 0;
    /// @brief Indicates that the primitive has user-defined output size (non-zero value).
    bool with_output_size;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief type of index output
    data_types index_element_type = data_types::i32;
    bool maxPoolOpset8Features{false};

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!indices_output.empty())
            ret.push_back(indices_output);
        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
