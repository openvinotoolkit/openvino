// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "common/primitive_hashing_utils.hpp"
#include "cpu_memory.h"
#include "executor.hpp"
#include "nodes/common/dnnl_executor.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/coordinate_diff.hpp"

namespace ov::intel_cpu {

struct DeconvAttrs {
    std::vector<ptrdiff_t> kernel;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    ov::CoordinateDiff outputPadding;
    bool withBiasesParam = false;
#if defined(OV_CPU_WITH_ACL)
    bool aclFastMath = false;
#endif
};

class DeconvExecutor {
public:
    explicit DeconvExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

    virtual bool init(const DeconvAttrs& deconvAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src,
                      const std::vector<MemoryPtr>& dst,
                      const void* post_ops_data_) = 0;
    virtual ~DeconvExecutor() = default;
    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;

protected:
    DeconvAttrs deconvAttrs;
    ExecutorContext::CPtr context;
};

using DeconvExecutorPtr = std::shared_ptr<DeconvExecutor>;
using DeconvExecutorCPtr = std::shared_ptr<const DeconvExecutor>;

class DeconvExecutorBuilder {
public:
    ~DeconvExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const DeconvAttrs& convAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual DeconvExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using DeconvExecutorBuilderPtr = std::shared_ptr<DeconvExecutorBuilder>;
using DeconvExecutorBuilderCPtr = std::shared_ptr<const DeconvExecutorBuilder>;

}  // namespace ov::intel_cpu
