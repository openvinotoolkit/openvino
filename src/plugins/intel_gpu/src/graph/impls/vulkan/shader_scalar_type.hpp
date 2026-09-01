// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_shader_abi.hpp"

namespace cldnn::vulkan {

inline bool is_supported_shader_scalar_type(data_types type) {
    return one_of(type,
                  {data_types::f16,
                   data_types::f32,
                   data_types::i8,
                   data_types::u8,
                   data_types::i16,
                   data_types::u16,
                   data_types::i32,
                   data_types::u32,
                   data_types::i64,
                   data_types::boolean});
}

inline bool is_integer_shader_scalar_type(data_types type) {
    return one_of(type,
                  {data_types::i8, data_types::u8, data_types::i16, data_types::u16, data_types::i32, data_types::u32, data_types::i64, data_types::boolean});
}

inline shader_abi::scalar_type to_shader_scalar_type(data_types type) {
    switch (type) {
    case data_types::f16:
        return shader_abi::scalar_type::f16;
    case data_types::f32:
        return shader_abi::scalar_type::f32;
    case data_types::i8:
        return shader_abi::scalar_type::i8;
    case data_types::u8:
        return shader_abi::scalar_type::u8;
    case data_types::i16:
        return shader_abi::scalar_type::i16;
    case data_types::u16:
        return shader_abi::scalar_type::u16;
    case data_types::i32:
        return shader_abi::scalar_type::i32;
    case data_types::u32:
        return shader_abi::scalar_type::u32;
    case data_types::i64:
        return shader_abi::scalar_type::i64;
    case data_types::boolean:
        return shader_abi::scalar_type::boolean;
    default:
        OPENVINO_THROW("[GPU][Vulkan] Unsupported shader scalar type ", ov::element::Type(type).get_type_name());
    }
}

}  // namespace cldnn::vulkan
