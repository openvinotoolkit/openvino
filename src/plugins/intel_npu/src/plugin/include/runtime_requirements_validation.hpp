// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/npu.hpp"

namespace intel_npu {

/**
 * @brief Looks for runtime requirements within @p arguments , evaluates them, and returns the result. Both runtime
 * requirements versions are supported by this function: the version introduced along with the HSM format, and the one
 * preceding it.
 *
 * @returns ov::CompatibilityCheck::NOT_APPLICABLE if:
 *   * The runtime requirements are not found within @p arguments
 *   * The runtime requirements are empty
 *   * The legacy format is used, and the compiler requirements are empty
 *   * The current format is used, and the final evaluation result is "NOT_APPLICABLE" (may happen if e.g. the compiler
 * requirements are empty)
 *
 * ov::CompatibilityCheck::UNSUPPORTED if:
 *   * The compatibility string parser failed to parse
 *   * The handler for either of the two versions failed to interpret the content
 *   * The final evaluation result (after querying the compiler as well) is "UNSUPPORTED".
 *
 * ov::CompatibilityCheck::SUPPORTED otherwise.
 */
ov::CompatibilityCheck validateCompatibilityDescriptor(
    const ov::AnyMap& arguments,
    const ov::SoPtr<IEngineBackend>& backend,
    const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper);

}  // namespace intel_npu
