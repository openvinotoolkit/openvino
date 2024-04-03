// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Factory

#pragma once

#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu_private_properties.hpp"  // AL
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

ov::SoPtr<ICompiler> createCompiler(ov::intel_npu::CompilerType compilerType, const Logger& log);

}  // namespace intel_npu
