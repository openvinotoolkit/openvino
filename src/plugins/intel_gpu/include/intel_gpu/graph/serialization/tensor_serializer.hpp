// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "intel_gpu/runtime/tensor.hpp"

namespace cldnn {
template <typename BufferType>
class Serializer<BufferType, tensor, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const tensor& tensor_obj) {
        buffer << tensor_obj.sizes();
    }
};

template <typename BufferType>
class Serializer<BufferType, tensor, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, tensor& tensor_obj) {
        std::vector<tensor::value_type> sizes;
        buffer >> sizes;
        tensor_obj = tensor(sizes);
    }
};

}  // namespace cldnn
