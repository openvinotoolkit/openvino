// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <vector>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "intel_gpu/runtime/layout.hpp"

namespace cldnn {
template <typename BufferType>
class Serializer<BufferType, cldnn::layout, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const cldnn::layout& _layout) {
        buffer << make_data(&_layout.data_type, sizeof(cldnn::data_types));
        buffer << make_data(&_layout.format, sizeof(cldnn::format));
        buffer << _layout.data_padding.filling_value();
        buffer << _layout.data_padding.lower_size().sizes();
        buffer << _layout.data_padding.upper_size().sizes();

        const std::vector<cldnn::tensor::value_type> _sizes = _layout.get_tensor().sizes(_layout.format);
        buffer << _sizes;

        // for (uint i = 0; i < _shape_size; ++i) {
        //      buffer << _shape[i];
        // }

        // buffer << make_data(vector.data(), static_cast<uint64_t>(vector.size() * sizeof(T)));
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::layout, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, cldnn::layout& _layout) {
        buffer >> make_data(&_layout.data_type, sizeof(cldnn::data_types));
        buffer >> make_data(&_layout.format, sizeof(cldnn::format));

        {
            float _filling_value;
            buffer >> _filling_value;
            std::vector<cldnn::tensor::value_type> _lower_size;
            buffer >> _lower_size;
            std::vector<cldnn::tensor::value_type> _upper_size;
            buffer >> _upper_size;
            _layout.data_padding = cldnn::padding(_lower_size, _upper_size, _filling_value);
        }

        // ov::Shape _shape;
        // buffer >> _shape;

        std::vector<cldnn::tensor::value_type> _sizes;
        buffer >> _sizes;

        // size_t _shape_size;
        // buffer >> _shape_size;
        // std::vector<size_t> _size;
        // for (uint i = 0; i < _shape_size; ++i) {
        //     size_t val;
        //     buffer >> val;
        //     _size.emplace_back(val);
        // }

        // std::vector<cldnn::tensor::value_type> _sizes(_shape.size());
        // for (size_t i = 0; i < _shape.size(); i++) {
        //     _sizes[i] = _shape[i];
        // }

        _layout.set_tensor(tensor(_layout.format, _sizes));

        // cldnn::layout new_layout(ov::PartialShape(_size), _data_type,

        // typename std::vector<T>::size_type vector_size = 0UL;
        // buffer >> vector_size;
        // vector.resize(vector_size);
        // buffer >> make_data(vector.data(), static_cast<uint64_t>(vector_size * sizeof(T)));
    }
};

}  // namespace cldnn