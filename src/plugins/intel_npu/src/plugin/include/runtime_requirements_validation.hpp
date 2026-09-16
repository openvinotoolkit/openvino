// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/npu.hpp"

namespace intel_npu {

/**
 * @brief Looks for runtime requirements within "arguments", evaluates them, and returns the result.
 */
ov::CompatibilityCheck validateCompatibilityDescriptor(
    const ov::AnyMap& arguments,
    const ov::SoPtr<IEngineBackend>& backend,
    const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper);

}  // namespace intel_npu
