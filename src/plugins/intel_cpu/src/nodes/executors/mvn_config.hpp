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
    std::vector<const void*> postOpsDataPtrs;
};
using MVNConfig = executor::Config<MVNAttrs>;

inline VectorDims transformTo5DCase(const VectorDims& shape, bool initAcrossChannels) {
    switch (shape.size()) {
    // for 1 and 2 rank, if initAcrossChannels_ is true, adjust shape to fully vectorize under the unified 5d procedure.
    // otherwise there are not enough data in spatial dimension to process in one kernel.
    case 1:  // C
        if (initAcrossChannels) {
            return VectorDims({1, 1, 1, 1, shape[0]});
        } else {
            return VectorDims({1, shape[0], 1, 1, 1});
        }
    case 2:  // NC
        if (initAcrossChannels) {
            return VectorDims({1, shape[0], 1, shape[1], 1});
        } else {
            return VectorDims({shape[0], shape[1], 1, 1, 1});
        }
    case 3: {
        return VectorDims({shape[0], shape[1], 1, shape[2], 1});
    }
    case 4: {
        return VectorDims({shape[0], shape[1], 1, shape[2], shape[3]});
    }
    case 5: {
        return VectorDims({shape[0], shape[1], shape[2], shape[3], shape[4]});
    }
    default: {
        OPENVINO_THROW("MVN executor doesn't support planar layout with rank: ", shape.size());
    }
    }
}

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