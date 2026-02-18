// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/permute_kernel.h"
#include "transpose_config.hpp"

namespace ov::intel_cpu {

class TransposeExecutor : public Executor {
public:
    static jit_permute_config_params prepareParams(const PermuteParams& params);
    TransposeExecutor(const TransposeAttrs& attrs, ExecutorContext::CPtr context);
    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    ~TransposeExecutor() override = default;

protected:
    virtual bool init(const MemoryArgs& memory) = 0;
    PermuteParams permuteParams;
    bool isInitialized = false;
    const ExecutorContext::CPtr context;
};
using TransposeExecutorPtr = std::shared_ptr<TransposeExecutor>;
using TransposeExecutorCPtr = std::shared_ptr<const TransposeExecutor>;

}  // namespace ov::intel_cpu
