// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_npu/common/dynamic_arguments.hpp"
#include "intel_npu/runtime/npu_vm_runtime.hpp"

namespace intel_npu {

/**
 * @brief Implementation of the opaque @c _impl carried by @ref DynamicMemRefType.
 * @details Wraps a VM-runtime MemRef handle and tracks whether the host-side description
 * changed since the last sync to the device. All @c npuVMRuntime* calls live in the
 * corresponding .cpp so the VM loader header is not transitively pulled into every TU
 * that includes this file.
 *
 * The type is shared between the compiler adapter (which seeds the initial bindings) and
 * the level-zero backend pipeline (which drives the VM execution).
 */
struct DynamicMemRefImpl {
    npu_vm_runtime_mem_ref_handle_t _memRef = nullptr;
    bool _ptrUpdated = false;
    bool _shapeUpdated = false;
    bool _strideUpdated = false;

    DynamicMemRefImpl() = default;
    ~DynamicMemRefImpl();

    /// Push the latest host-side description into the underlying VM MemRef and record what changed.
    void updateMemRefHandleStatus(DynamicMemRefType& memref);

    /// Read the description currently stored in the VM MemRef back into @p memref.
    void alignWithHandle(DynamicMemRefType& memref);

private:
    void createMemRef(int64_t dimsCount);
    void destroyMemRef();
};

/**
 * @brief Implementation of the opaque @c _impl carried by @ref DynamicMemRefType.
 * @details Owns the VM execution context and the flattened MemRef-handle arrays used by
 * @c npuVMRuntimeExecute. The execution context is created lazily via
 * @ref ensureExecutionContext on the first execute call and destroyed when this struct is
 * destroyed -- Create/Destroy of the context are paired in the same translation unit.
 */
struct DynamicArgumentsImpl {
    std::vector<npu_vm_runtime_mem_ref_handle_t> _inputMemRefs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _outputMemRefs;
    npu_vm_runtime_execute_params_t _executeParams = {};

    DynamicArgumentsImpl() = default;
    ~DynamicArgumentsImpl();

    /// Lazily create the VM execution context for @p vmRuntime. No-op if already created.
    void ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime);
};

}  // namespace intel_npu
