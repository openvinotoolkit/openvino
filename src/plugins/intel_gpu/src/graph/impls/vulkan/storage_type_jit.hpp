// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "kernel_selector/jitter.h"
#include "vulkan/vulkan_device.hpp"

namespace cldnn::vulkan {

// Keep the shared kernel's storage types and expressions. On storage-only
// devices, qualify its existing scalar conversion/accumulator hooks so LLVM
// cannot fold the promoted arithmetic back to unsupported 8/16-bit operations.
inline kernel_selector::JitConstants storage_type_jit(const kernel_selector::JitConstants& jit,
                                                     const vulkan_device& device) {
    const auto wide_type = [&](const std::string& type) -> std::string {
        if ((type == "char" || type == "uchar") && !device.supports_arithmetic_type(data_types::i8)) {
            return "int";
        }
        if ((type == "short" || type == "ushort") && !device.supports_arithmetic_type(data_types::i16)) {
            return "int";
        }
        if (type == "half" && !device.supports_arithmetic_type(data_types::f16)) {
            return "float";
        }
        return {};
    };
    auto definitions = jit.GetDefinitions();
    std::map<std::string, std::string> values(definitions.begin(), definitions.end());
    bool needs_storage_types = false;
    for (const auto& definition : definitions) {
        if ((definition.first.rfind("INPUT", 0) == 0 || definition.first.rfind("OUTPUT", 0) == 0) &&
            !wide_type(definition.second).empty()) {
            needs_storage_types = true;
            break;
        }
    }
    if (!needs_storage_types) {
        return jit;
    }

    const auto temporary = [](const std::string& type, const std::string& value) {
        // A qualified compound literal is an OpenCL C lvalue, unlike a cast
        // whose volatile qualifier is discarded. It does not change the buffer ABI.
        return "((volatile " + type + "){" + value + "})";
    };
    kernel_selector::JitConstants result{};
    for (auto& definition : definitions) {
        const auto& name = definition.first;
        auto& value = definition.second;
        if (name == "ACCUMULATOR_TYPE") {
            const auto promoted = wide_type(value);
            if (!promoted.empty()) {
                value = promoted;
            }
            if (value == "int" || value == "uint" || value == "float") {
                value = "volatile " + value;
            }
        } else if (name.rfind("DECODE_INPUT", 0) == 0 && name.find("_COMPUTE_TYPE(v)") != std::string::npos) {
            const auto prefix = name.substr(std::string("DECODE_").size(), name.find("_COMPUTE_TYPE(v)") - std::string("DECODE_").size());
            auto promoted = wide_type(values[prefix + "_TYPE"]);
            if (!promoted.empty()) {
                if (values[prefix + "_IS_BF16"] == "1") {
                    promoted = "float";
                }
                value = temporary(promoted, value);
            }
        } else if (name.rfind("TO_", 0) == 0 &&
                   (name.find("_TYPE(v)") != std::string::npos || name.find("_TYPE_SAT(v)") != std::string::npos)) {
            for (const std::string type : {"char", "uchar", "short", "ushort"}) {
                const auto conversion = "convert_" + type;
                if (wide_type(type).empty() || value.rfind(conversion, 0) != 0 ||
                    value.size() < 3 || value.substr(value.size() - 3) != "(v)") {
                    continue;
                }
                const auto function = value.substr(0, value.size() - 3);
                const auto suffix = function.substr(conversion.size());
                // Preserve the shared conversion's rounding/saturation modifiers.
                value = function + "(" + temporary("int", "convert_int" + suffix + "(v)") + ")";
                break;
            }
        }
        result.AddConstant(kernel_selector::MakeJitConstant(name, value));
    }
    return result;
}

template <typename Kernel, typename Params>
class StorageTypeKernel final : public Kernel {
public:
    explicit StorageTypeKernel(const vulkan_device& device) : _device(device) {}

    kernel_selector::JitConstants GetJitConstants(const Params& params) const override {
        return storage_type_jit(Kernel::GetJitConstants(params), _device);
    }

private:
    const vulkan_device& _device;
};

}  // namespace cldnn::vulkan
