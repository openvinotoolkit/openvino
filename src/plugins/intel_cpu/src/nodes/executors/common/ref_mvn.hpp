// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "cpu_types.h"
#include "post_ops.hpp"

namespace ov::intel_cpu {

class MVNRefExecutor : public Executor {
public:
    MVNRefExecutor(MVNAttrs mvnAttrs, MemoryArgs memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override {
        memoryArgs = memory;
        return true;
    }

    void execute() override {
        executeImpl(memoryArgs);
    }

    void execute(const MemoryArgs& memory) override {
        executeImpl(memory);
    }

    void executeImpl(const MemoryArgs& memory);

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

    static bool supports(const MVNConfig& config);

private:
    MVNAttrs attrs;
    MemoryArgs memoryArgs;
    const ExecutorContext::CPtr context;
    size_t src_data_size = 0;
    size_t dst_data_size = 0;
    VectorDims shape5D;
};

}  // namespace ov::intel_cpu
