// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "shl.hpp"

namespace ov::intel_cpu {

class ShlEltwiseExecutor : public Executor {
public:
    explicit ShlEltwiseExecutor(EltwiseAttrs attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);
    static bool supports(const EltwiseConfig& config);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::shl;
    }

private:
    bool init(const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs);

    EltwiseAttrs shlEltwiseAttrs;
    ShlSession sess;
    std::vector<ShlTensor> srcTensors, dstTensors;
    std::unique_ptr<IShlParams> params;
    std::function<int()> shlExecFunc;
};

}  // namespace ov::intel_cpu
