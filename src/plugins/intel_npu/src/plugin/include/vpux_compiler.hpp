//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Compiler Factory

#pragma once

#include "npu/utils/logger/logger.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "vpux/al/icompiler.hpp"
#include "vpux_private_properties.hpp"  // AL

namespace vpux {

ov::SoPtr<ICompiler> createCompiler(ov::intel_npu::CompilerType compilerType, const intel_npu::Logger& log);

}  // namespace vpux
