// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "cpu_types.h"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

class IEltwiseExecutor {
public:
    IEltwiseExecutor() = default;
    virtual ~IEltwiseExecutor() = default;

    virtual void exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) = 0;
    [[nodiscard]] virtual size_t getBatchDimIdx() const = 0;
    [[nodiscard]] virtual const VectorDims& getOutDims() const = 0;
    [[nodiscard]] virtual impl_desc_type implType() const = 0;
    // // Executor interface
    // void execute(const MemoryArgs& /*memory*/) override {
    //     // Convert MemoryArgs to jit_eltwise_call_args_ptrs and call exec
    //     // This will be implemented in each specific executor
    //     throw std::runtime_error("execute(MemoryArgs) should be implemented in derived classes");
    // }

    // bool update(const MemoryArgs& /*memory*/) override {
    //     // Default implementation - always return true for stateless executors
    //     return true;
    // }
};

using EltwiseExecutorPtr = std::shared_ptr<IEltwiseExecutor>;

}  // namespace ov::intel_cpu
