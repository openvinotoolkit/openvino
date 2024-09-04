// Copyright (C) 2018-2024 Intel Corporation
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
class Serializer<BufferType, cldnn::format_traits, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const cldnn::format_traits& traits) {
        buffer << traits.str;
        buffer << traits.batch_num;
        buffer << traits.feature_num;
        buffer << traits.spatial_num;
        buffer << traits.group_num;
        buffer << traits._order;
        buffer << traits.order;
        buffer << traits.internal_order;
        buffer << traits.block_sizes.size();
        for (auto& block_size : traits.block_sizes) {
            buffer << block_size.first;
            buffer << block_size.second;
        }
        for (auto& block_size : traits.logic_block_sizes) {
            buffer << block_size.first;
            buffer << block_size.second;
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::format_traits, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, cldnn::format_traits& traits) {
        buffer >> traits.str;
        buffer >> traits.batch_num;
        buffer >> traits.feature_num;
        buffer >> traits.spatial_num;
        buffer >> traits.group_num;
        buffer >> traits._order;
        buffer >> traits.order;
        buffer >> traits.internal_order;

        size_t num_block_size;
        buffer >> num_block_size;
        size_t blk_size;
        int axis_idx;
        for (size_t i = 0; i < num_block_size; i++) {
            buffer >> blk_size;
            buffer >> axis_idx;
            traits.block_sizes.push_back(std::make_pair(blk_size, axis_idx));
        }
        for (size_t i = 0; i < num_block_size; i++) {
            buffer >> blk_size;
            buffer >> axis_idx;
            traits.logic_block_sizes.push_back(std::make_pair(blk_size, axis_idx));
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::format, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const cldnn::format& format) {
        cldnn::format::type fmt_type = format;
        buffer << make_data(&fmt_type, sizeof(cldnn::format::type));
        if (fmt_type == cldnn::format::custom)
            buffer << format.traits();
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::format, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, cldnn::format& format) {
        cldnn::format::type fmt_type = cldnn::format::type::any;
        buffer >> make_data(&fmt_type, sizeof(cldnn::format::type));
        if (fmt_type == cldnn::format::custom) {
            cldnn::format_traits traits;
            buffer >> traits;
            format = cldnn::format(traits);
        } else {
            format = cldnn::format(fmt_type);
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::layout, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const cldnn::layout& _layout) {
        buffer << make_data(&_layout.data_type, sizeof(cldnn::data_types));
        buffer << _layout.format;
        buffer << _layout.data_padding;
        buffer << _layout.get_partial_shape();
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::layout, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, cldnn::layout& _layout) {
        buffer >> make_data(&_layout.data_type, sizeof(cldnn::data_types));
        buffer >> _layout.format;
        buffer >> _layout.data_padding;

        ov::PartialShape partial_shape;
        buffer >> partial_shape;
        _layout.set_partial_shape(partial_shape);
    }
};

}  // namespace cldnn
