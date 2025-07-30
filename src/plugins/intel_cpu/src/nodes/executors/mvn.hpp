// Copyright (C) 2018-2025 Intel Corporation
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
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

enum MVNLayoutType : uint8_t { mvn_planar, mvn_block, mvn_by_channel };

// Defines way to add epsilon: inside sqrt or outside.
enum MVNEpsMode : uint8_t { INSIDE_SQRT, OUTSIDE_SQRT };

struct MVNAttrs {
    MVNLayoutType layout = mvn_planar;
    bool initAcrossChannels_ = false;
    bool execAcrossChannels_ = false;
    bool normalizeVariance_ = false;
    float epsValue_ = 0.0F;
    MVNEpsMode epsMode_ = INSIDE_SQRT;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
};

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

class MVNExecutorBuilder {
public:
    virtual ~MVNExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const MVNAttrs& mvnAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual MVNExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using MVNExecutorBuilderPtr = std::shared_ptr<MVNExecutorBuilder>;
using MVNExecutorBuilderCPtr = std::shared_ptr<const MVNExecutorBuilder>;

}  // namespace ov::intel_cpu
