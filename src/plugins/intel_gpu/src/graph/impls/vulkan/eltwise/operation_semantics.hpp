// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "../eltwise_shader_abi.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "openvino/core/except.hpp"

namespace cldnn::vulkan::eltwise_detail {

inline constexpr std::optional<eltwise_shader_abi::mode> try_shader_mode_code(eltwise_mode mode) noexcept {
    switch (mode) {
    case eltwise_mode::sum:
        return eltwise_shader_abi::mode::sum;
    case eltwise_mode::sub:
        return eltwise_shader_abi::mode::sub;
    case eltwise_mode::max:
        return eltwise_shader_abi::mode::max;
    case eltwise_mode::prod:
        return eltwise_shader_abi::mode::prod;
    case eltwise_mode::div:
        return eltwise_shader_abi::mode::div;
    case eltwise_mode::min:
        return eltwise_shader_abi::mode::min;
    case eltwise_mode::pow:
        return eltwise_shader_abi::mode::pow;
    case eltwise_mode::squared_diff:
        return eltwise_shader_abi::mode::squared_diff;
    case eltwise_mode::mod:
        return eltwise_shader_abi::mode::mod;
    case eltwise_mode::eq:
        return eltwise_shader_abi::mode::eq;
    case eltwise_mode::ne:
        return eltwise_shader_abi::mode::ne;
    case eltwise_mode::lt:
        return eltwise_shader_abi::mode::lt;
    case eltwise_mode::le:
        return eltwise_shader_abi::mode::le;
    case eltwise_mode::gt:
        return eltwise_shader_abi::mode::gt;
    case eltwise_mode::ge:
        return eltwise_shader_abi::mode::ge;
    case eltwise_mode::logic_and:
        return eltwise_shader_abi::mode::logic_and;
    case eltwise_mode::logic_or:
        return eltwise_shader_abi::mode::logic_or;
    case eltwise_mode::logic_xor:
        return eltwise_shader_abi::mode::logic_xor;
    case eltwise_mode::floor_mod:
        return eltwise_shader_abi::mode::floor_mod;
    case eltwise_mode::is_finite:
        return eltwise_shader_abi::mode::is_finite;
    case eltwise_mode::is_inf:
        return eltwise_shader_abi::mode::is_inf;
    case eltwise_mode::is_nan:
        return eltwise_shader_abi::mode::is_nan;
    case eltwise_mode::right_shift:
        return eltwise_shader_abi::mode::right_shift;
    case eltwise_mode::left_shift:
        return eltwise_shader_abi::mode::left_shift;
    case eltwise_mode::bitwise_and:
        return eltwise_shader_abi::mode::bitwise_and;
    case eltwise_mode::bitwise_or:
        return eltwise_shader_abi::mode::bitwise_or;
    case eltwise_mode::bitwise_xor:
        return eltwise_shader_abi::mode::bitwise_xor;
    case eltwise_mode::atan2:
        return eltwise_shader_abi::mode::atan2;
    default:
        return std::nullopt;
    }
}

inline constexpr bool is_supported_mode(eltwise_mode mode) noexcept {
    return try_shader_mode_code(mode).has_value();
}

inline constexpr bool is_unary_mode(eltwise_mode mode) noexcept {
    switch (mode) {
    case eltwise_mode::is_finite:
    case eltwise_mode::is_inf:
    case eltwise_mode::is_nan:
        return true;
    default:
        return false;
    }
}

inline constexpr bool is_fused_mode(eltwise_mode mode) noexcept {
    switch (mode) {
    case eltwise_mode::sum:
    case eltwise_mode::prod:
    case eltwise_mode::sub:
    case eltwise_mode::div:
        return true;
    default:
        return false;
    }
}

inline constexpr bool is_bitwise_mode(eltwise_mode mode) noexcept {
    switch (mode) {
    case eltwise_mode::right_shift:
    case eltwise_mode::left_shift:
    case eltwise_mode::bitwise_and:
    case eltwise_mode::bitwise_or:
    case eltwise_mode::bitwise_xor:
        return true;
    default:
        return false;
    }
}

inline eltwise_shader_abi::mode shader_mode_code(eltwise_mode mode) {
    const auto shader_mode = try_shader_mode_code(mode);
    if (!shader_mode.has_value()) {
        OPENVINO_THROW("[GPU][Vulkan] Unsupported Eltwise shader mode");
    }
    return *shader_mode;
}

}  // namespace cldnn::vulkan::eltwise_detail
