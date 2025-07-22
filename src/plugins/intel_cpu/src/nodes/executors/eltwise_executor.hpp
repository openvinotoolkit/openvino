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
};

using EltwiseExecutorPtr = std::shared_ptr<IEltwiseExecutor>;

}  // namespace ov::intel_cpu
