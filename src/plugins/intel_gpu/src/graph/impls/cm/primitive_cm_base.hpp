// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "impls/ocl_v2/primitive_ocl_base.hpp"

namespace ov::intel_gpu::cm {

// No changes are needed from implementation standpoint, so just use alias in case if we need some changes in the future
using PrimitiveImplCM = ov::intel_gpu::ocl::PrimitiveImplOCL;
using Stage = ov::intel_gpu::ocl::Stage;

template <typename T>
JitConstant make_jit_constant(const std::string& name, T value) {
    if constexpr (std::is_floating_point<T>::value) {
        return JitConstant(name, std::to_string(value));
    } else {
        return ov::intel_gpu::make_jit_constant(name, value);
    }
}

template <typename T>
JitConstant make_jit_constant(const JitTerm& name, T value) {
    return make_jit_constant(name.str(), value);
}

}  // namespace ov::intel_gpu::cm
