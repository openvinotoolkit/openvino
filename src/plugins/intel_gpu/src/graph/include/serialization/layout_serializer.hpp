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

        std::vector<cldnn::tensor::value_type> _sizes = _layout.get_tensor().sizes(_layout.format);
        // Temp WA for bs_x_bsv16
        if (_layout.format == cldnn::format::bs_x_bsv16) {
            std::vector<cldnn::tensor::value_type> _tmp_sizes = _layout.get_tensor().sizes();
            _sizes[0] = _tmp_sizes[0];
            _sizes[1] = _tmp_sizes[1];
        }
        buffer << _sizes;
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

        std::vector<cldnn::tensor::value_type> _sizes;
        buffer >> _sizes;

        // Temp WA for bs_x_bsv16
        if (_layout.format == cldnn::format::bs_x_bsv16) {
            _layout.set_tensor(tensor(_sizes));
        } else {
            _layout.set_tensor(tensor(_layout.format, _sizes));
        }
    }
};

}  // namespace cldnn