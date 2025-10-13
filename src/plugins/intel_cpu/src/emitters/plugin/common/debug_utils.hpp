// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef CPU_DEBUG_CAPS

#    include <cstddef>
#    include <cstdint>
#    include <iostream>
#    include <sstream>
#    include <type_traits>

namespace ov::intel_cpu::debug_utils {

namespace detail {

template <typename T>
void format_value(std::stringstream& ss, const T& value) {
    if constexpr (std::is_floating_point_v<T>) {
        ss << value;
    } else if constexpr (std::is_signed_v<T>) {
        ss << static_cast<int64_t>(value);
    } else {
        ss << static_cast<uint64_t>(value);
    }
}

template <typename T>
void print_values_impl(const char* name, const char* orig_name, T* ptr, size_t count) {
    std::stringstream ss;
    if (name) {
        ss << name << " | ";
    }
    ss << orig_name << ": ";

    if (count == 1) {
        // Single value (scalar register)
        format_value(ss, *ptr);
    } else {
        // Multiple values (vector register)
        ss << "{";
        for (size_t idx = 0; idx < count; ++idx) {
            if (idx != 0) {
                ss << ", ";
            }
            format_value(ss, ptr[idx]);
        }
        ss << "}";
    }
    ss << '\n';
    std::cout << ss.str();
}

}  // namespace detail

template <typename T>
void print_reg_prc(const char* name, const char* orig_name, T* ptr) {
    detail::print_values_impl(name, orig_name, ptr, 1);
}

template <typename PRC_T, size_t vlen>
void print_vmm_prc(const char* name, const char* orig_name, PRC_T* ptr) {
    constexpr size_t elems = vlen / sizeof(PRC_T);
    detail::print_values_impl(name, orig_name, ptr, elems);
}

}  // namespace ov::intel_cpu::debug_utils

#endif  // CPU_DEBUG_CAPS
