// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/common/dnnl_executor.h"

namespace ov::intel_cpu {

class IMatmulExecutor {
public:
    virtual void exec(const std::unordered_map<int, dnnl::memory>& primArgs, const dnnl::stream& strm) = 0;
    [[nodiscard]] virtual DnnlMemoryDescPtr getScratchPadDesc() const = 0;
    virtual ~IMatmulExecutor() = default;
};

class DnnlMatmulExecutor : public IMatmulExecutor {
public:
    explicit DnnlMatmulExecutor(const dnnl::primitive_desc& pd) : m_executor(pd) {}
    void exec(const std::unordered_map<int, dnnl::memory>& primArgs, const dnnl::stream& strm) override {
        m_executor.exec(primArgs, strm);
    }
    DnnlMemoryDescPtr getScratchPadDesc() const override {
        return m_executor.getScratchPadDesc();
    }

private:
    DnnlExecutorLegacy m_executor;
};

}  // namespace ov::intel_cpu
