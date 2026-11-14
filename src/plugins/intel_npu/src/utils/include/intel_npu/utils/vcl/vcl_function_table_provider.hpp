// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

namespace intel_npu {

struct VCLFunctionTable;

/**
 * @brief Obtains the VCL entry points on demand.
 *
 * Providers defer loading until the entry points are requested to resolve or construct a compiler-in-plugin,
 * or to create a profiling decoder. Profiling imports can request them even when using the driver compiler.
 * A provider should remember a failed load, so that callers do not repeat it, and always supply the same set of
 * entry points: what is built from them, such as a CompilerAdapterFactory, is bound to that set.
 *
 * Lives apart from vcl_api.hpp so that headers that only pass a provider around do not pull in the VCL C API.
 */
using VCLFunctionTableProvider = std::function<std::shared_ptr<const VCLFunctionTable>()>;

}  // namespace intel_npu
