// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "config.h"
#include "cpu_types.h"
#include "executor_config.hpp"
#include "nodes/executors/executor.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

enum MVNLayoutType : uint8_t { mvn_planar, mvn_block, mvn_by_channel };

// Defines way to add epsilon: inside sqrt or outside.
enum MVNEpsMode : uint8_t { INSIDE_SQRT, OUTSIDE_SQRT };

struct MVNAttrs {
    MVNLayoutType layout = mvn_planar;
    bool initAcrossChannels_ = false;
    bool execAcrossChannels_ = false;
    bool normalizeVariance_ = false;
    float epsValue_ = 0.0f;
    MVNEpsMode epsMode_ = INSIDE_SQRT;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
};
using MVNConfig = executor::Config<MVNAttrs>;

namespace legacy {

class MVNExecutor {
public:
    MVNExecutor(ExecutorContext::CPtr context);
    virtual bool init(const MVNAttrs& mvnAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src,
                      const std::vector<MemoryPtr>& dst,
                      const void* post_ops_data_) = 0;
    virtual ~MVNExecutor() = default;

    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;

    static VectorDims transformTo5DCase(const VectorDims& shape, bool initAcrossChannels);

protected:
    MVNAttrs mvnAttrs;
    const ExecutorContext::CPtr context;
};

using MVNExecutorPtr = std::shared_ptr<MVNExecutor>;
using MVNExecutorCPtr = std::shared_ptr<const MVNExecutor>;

class MVNExecutorBase {
public:
    MVNExecutorBase(const MVNAttrs& mvnAttrs);
    virtual void exec(const uint8_t* in_ptr_,
                      uint8_t* dst_data,
                      const void* post_ops_data_,
                      const VectorDims& shape5d) = 0;
    virtual ~MVNExecutorBase() = default;

protected:
    MVNAttrs mvnAttrs;
    size_t src_data_size = 0;
    size_t dst_data_size = 0;
};

}  // namespace legacy

}  // namespace ov::intel_cpu