// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/normalize.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

class ACLNormalizeL2Executor : public NormalizeL2Executor {
public:
    explicit ACLNormalizeL2Executor(const ExecutorContext::CPtr context);

    bool init(const NormalizeL2Attrs& normalizeL2Attrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void **post_ops_data_) override;
    impl_desc_type getImplType() const override { return implType; }

private:
    NormalizeL2Attrs aclNormalizeL2Attrs;
    impl_desc_type implType = impl_desc_type::acl;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEL2NormalizeLayer> aclNEL2NormalizeLayer = nullptr;
};

class ACLNormalizeL2ExecutorBuilder : public NormalizeL2ExecutorBuilder {
public:
    bool isSupported(const NormalizeL2Attrs& normalizeL2Attrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (normalizeL2Attrs.isFusing) return false;
        // Unsupported EpsMode::ADD of NormalizeL2 layer.
        if (normalizeL2Attrs.epsMode == NormEpsMode::ADD) return false;
        return true;
    }

    NormalizeL2ExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLNormalizeL2Executor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov