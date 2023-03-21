// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include "buffer.hpp"

namespace cldnn {
struct input_info;

template <typename BufferType>
class Serializer<BufferType, input_info, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const input_info& input) {
        buffer << input.pid;
        buffer << input.idx;
    }
};

template <typename BufferType>
class Serializer<BufferType, input_info, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, input_info& input) {
        buffer >> input.pid;
        buffer >> input.idx;
    }
};

}  // namespace cldnn
