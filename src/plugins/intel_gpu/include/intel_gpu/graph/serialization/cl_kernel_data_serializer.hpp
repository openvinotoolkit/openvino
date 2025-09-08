// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include "buffer.hpp"
#include "kernel_selector_common.h"


namespace cldnn {

template <typename BufferType>
class Serializer<BufferType, kernel_selector::clKernelData, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const kernel_selector::clKernelData& data) {
        data.save(buffer);
    }
};

template <typename BufferType>
class Serializer<BufferType, kernel_selector::clKernelData, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, kernel_selector::clKernelData& data) {
        data.load(buffer);
    }
};

}  // namespace cldnn
