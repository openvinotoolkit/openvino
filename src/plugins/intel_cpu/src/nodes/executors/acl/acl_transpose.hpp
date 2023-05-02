// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/transpose.hpp"

namespace ov {
namespace intel_cpu {

class ACLTransposeExecutor : public TransposeExecutor {
public:
    explicit ACLTransposeExecutor(const ExecutorContext::CPtr context);
    bool init(const TransposeParams& transposeParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) override;
    impl_desc_type getImplType() const override { return implType; }
private:
    impl_desc_type implType = impl_desc_type::acl;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEPermute> acl_permute;
};

class ACLTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    bool isSupported(const TransposeParams& transposeParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        for (const auto &srcD : srcDescs) {
            for (const auto &dstD : dstDescs) {
                if (!(srcD->hasLayoutType(LayoutType::ncsp) &&
                      dstD->hasLayoutType(LayoutType::ncsp)) &&
                    !(srcD->hasLayoutType(LayoutType::nspc) &&
                      dstD->hasLayoutType(LayoutType::nspc)))
                    return false;
            }
        }
        return true;
    }

    TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLTransposeExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov