// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

enum PadMode : uint8_t { CONSTANT = 0, EDGE = 1, REFLECT = 2, SYMMETRIC = 3 };

struct PadAttrs {
    PadMode padMode = CONSTANT;
    float padValue = 0.F;
    std::vector<int32_t> padsBegin;
    std::vector<int32_t> padsEnd;
    int beginPadIdx = 0;
    int endPadIdx = 0;
    ov::element::Type prc;
    bool constPadValue = false;
    std::shared_ptr<CpuParallel> cpuParallel;
};

class PadExecutor {
public:
    explicit PadExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

    virtual bool init(const PadAttrs& padAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src,
                      const std::vector<MemoryPtr>& dst,
                      const void* post_ops_data_) = 0;

    virtual ~PadExecutor() = default;

    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;

protected:
    PadAttrs padAttrs;
    const ExecutorContext::CPtr context;
};

using PadExecutorPtr = std::shared_ptr<PadExecutor>;
using PadExecutorCPtr = std::shared_ptr<const PadExecutor>;

class PadExecutorBuilder {
public:
    virtual ~PadExecutorBuilder() = default;

    [[nodiscard]] virtual bool isSupported(const PadAttrs& padAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;

    [[nodiscard]] virtual PadExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using PadExecutorBuilderPtr = std::shared_ptr<PadExecutorBuilder>;
using PadExecutorBuilderCPtr = std::shared_ptr<const PadExecutorBuilder>;

}  // namespace ov::intel_cpu