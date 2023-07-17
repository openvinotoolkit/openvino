// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"

namespace cldnn {
enum class activation_func;

template <typename BufferType>
class Serializer<BufferType, activation_func, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const activation_func& activation) {
        buffer << make_data(&activation, sizeof(activation_func));
    }
};

template <typename BufferType>
class Serializer<BufferType, activation_func, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, activation_func& activation) {
        buffer >> make_data(&activation, sizeof(activation_func));
    }
};

}  // namespace cldnn
