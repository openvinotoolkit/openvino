// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <kernel_selector_utils.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kernel_selector {
namespace slice_scatter_utils {

static constexpr size_t MAX_SUPPORTED_DIM = 5;
static constexpr char JIT_AXES_BUFF_SIZE_NAME[] = "AXES_BUFFER_SIZE";

// Generates macros:
// - name_BUFFER
// - name_VAL0, name_VAL1 ...
inline void addJitConstantsForParam(JitConstants& jit,
                                    const std::string& name,
                                    const std::vector<std::int64_t>& compile_time_param,
                                    Datatype type,
                                    const std::function<std::string(std::string, size_t)>& dynamic_access_decorator) {
    const std::string BUFF_CONST_NAME = name + "_BUFFER";
    const std::string BUFF_PTR_NAME = name + "_buffer_ptr";
    const auto jit_name_decorator = [](std::string name, size_t i) {
        return name + "_VAL" + std::to_string(i);
    };

    if (compile_time_param.empty()) {
        // Dynamic param:
        const std::string type_str = toCLType(type);
        jit.AddConstant(
            MakeJitConstant(BUFF_CONST_NAME, "__global const " + type_str + "* restrict " + BUFF_PTR_NAME + ","));

        for (size_t i = 0; i < MAX_SUPPORTED_DIM; ++i) {
            const std::string i_str = std::to_string(i);
            const std::string jit_name = jit_name_decorator(name, i);
            const std::string access_str = dynamic_access_decorator(BUFF_PTR_NAME, i);
            jit.AddConstant(
                MakeJitConstant(jit_name, i_str + " < " + JIT_AXES_BUFF_SIZE_NAME + " ? (" + access_str + ") : -1"));
        }
    } else {
        // Static param:
        jit.AddConstant(MakeJitConstant(BUFF_CONST_NAME, ""));
        for (size_t i = 0; i < MAX_SUPPORTED_DIM; ++i) {
            const std::string jit_name = jit_name_decorator(name, i);
            const int64_t val = i < compile_time_param.size() ? compile_time_param[i] : -1;
            jit.AddConstant(MakeJitConstant(jit_name, val));
        }
    }
}

}  // namespace slice_scatter_utils
}  // namespace kernel_selector
