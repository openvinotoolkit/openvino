// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <vector>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "intel_gpu/primitives/loop.hpp"

namespace cldnn {
template <typename BufferType>
class Serializer<BufferType, cldnn::loop::io_primitive_map, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const cldnn::loop::io_primitive_map& io_primitive_map) {
        buffer << io_primitive_map.external_id;
        buffer << io_primitive_map.internal_id;
        buffer << io_primitive_map.axis;
        buffer << io_primitive_map.start;
        buffer << io_primitive_map.end;
        buffer << io_primitive_map.stride;
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::loop::io_primitive_map, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, cldnn::loop::io_primitive_map& io_primitive_map) {
        buffer >> io_primitive_map.external_id;
        buffer >> io_primitive_map.internal_id;
        buffer >> io_primitive_map.axis;
        buffer >> io_primitive_map.start;
        buffer >> io_primitive_map.end;
        buffer >> io_primitive_map.stride;
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::loop::backedge_mapping, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const cldnn::loop::backedge_mapping& backedge_mapping) {
        buffer << backedge_mapping.from;
        buffer << backedge_mapping.to;
    }
};

template <typename BufferType>
class Serializer<BufferType, cldnn::loop::backedge_mapping, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, cldnn::loop::backedge_mapping& backedge_mapping) {
        buffer >> backedge_mapping.from;
        buffer >> backedge_mapping.to;
    }
};
}  // namespace cldnn
