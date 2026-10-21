// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/format.hpp"
#include "npu_bytecode_utils/macro.hpp"

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ov {

namespace intel_npu::vm {

namespace detail {

[[noreturn]] void throwRuntimeError(const char* file, int line, const std::string& message);

template <typename T>
std::string toString(const T& value) {
    using ValueType = std::decay_t<T>;
    if constexpr (std::is_same_v<ValueType, std::string>) {
        return value;
    } else if constexpr (std::is_same_v<ValueType, std::string_view>) {
        return std::string(value);
    } else if constexpr (std::is_same_v<ValueType, const char*> || std::is_same_v<ValueType, char*>) {
        const char* strValue = value;
        return strValue == nullptr ? std::string{} : std::string(strValue);
    } else {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    }
}

template <typename... Args>
std::string formatMessage(std::string_view format, const Args&... args) {
    const std::vector<std::string> serializedArgs{toString(args)...};
    return formatString(format, serializedArgs);
}

}  // namespace detail

}  // namespace intel_npu::vm

}  // namespace ov

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

#define NPU_VM_THROW(...) \
    ov::intel_npu::vm::detail::throwRuntimeError(__FILE__, __LINE__, ov::intel_npu::vm::detail::formatMessage(__VA_ARGS__))

#define NPU_VM_THROW_WHEN(condition, ...) \
    if (NPU_VM_UNLIKELY(condition)) {     \
        NPU_VM_THROW(__VA_ARGS__);        \
    }

#define NPU_VM_THROW_UNLESS(condition, ...) \
    if (NPU_VM_UNLIKELY(!(condition))) {    \
        NPU_VM_THROW(__VA_ARGS__);          \
    }

// NOLINTEND(cppcoreguidelines-macro-usage)
