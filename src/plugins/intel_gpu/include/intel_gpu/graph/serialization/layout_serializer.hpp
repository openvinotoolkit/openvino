// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "intel_gpu/runtime/layout.hpp"

namespace cldnn {
template <typename BufferType>
class Serializer<BufferType, ov::PartialShape, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const ov::PartialShape& partial_shape) {
        std::vector<ov::Dimension> dimensions(partial_shape);
        buffer << dimensions.size();
        for (const auto& dimension : dimensions) {
            buffer << dimension.get_interval().get_min_val();
            buffer << dimension.get_interval().get_max_val();
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::PartialShape, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, ov::PartialShape& partial_shape) {
        size_t num_dimensions;
        buffer >> num_dimensions;
        for (size_t i = 0; i < num_dimensions; i++) {
            ov::Interval::value_type min_val, max_val;
            buffer >> min_val >> max_val;
            partial_shape.push_back(ov::Dimension(min_val, max_val));
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::layout, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const cldnn::layout& _layout) {
        buffer << make_data(&_layout.data_type, sizeof(cldnn::data_types));
        buffer << make_data(&_layout.format, sizeof(cldnn::format));
        buffer << _layout.data_padding;
        // buffer << _layout.data_padding.filling_value();
        // buffer << _layout.data_padding.lower_size().sizes();
        // buffer << _layout.data_padding.upper_size().sizes();
        buffer << _layout.get_partial_shape();
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::layout, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, cldnn::layout& _layout) {
        buffer >> make_data(&_layout.data_type, sizeof(cldnn::data_types));
        buffer >> make_data(&_layout.format, sizeof(cldnn::format));

        // {
        //     float _filling_value;
        //     buffer >> _filling_value;
        //     std::vector<cldnn::tensor::value_type> _lower_size;
        //     buffer >> _lower_size;
        //     std::vector<cldnn::tensor::value_type> _upper_size;
        //     buffer >> _upper_size;
        //     _layout.data_padding = cldnn::padding(_lower_size, _upper_size, _filling_value);
        // }
        buffer >> _layout.data_padding;

        ov::PartialShape partial_shape;
        buffer >> partial_shape;
        _layout.set_partial_shape(partial_shape);
    }
};

}  // namespace cldnn
