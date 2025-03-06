// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"
#include "acl_utils.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov::intel_cpu {

class AclEltwiseExecutor : public EltwiseExecutor {
public:
    explicit AclEltwiseExecutor(const ExecutorContext::CPtr context);
    static bool isEltwiseAlgorithmSupported(Algorithm algorithm);

    bool init(const EltwiseAttrs& attrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override {
        return implType;
    }

private:
    EltwiseAttrs aclEltwiseAttrs{};
    impl_desc_type implType = impl_desc_type::acl;
    std::vector<arm_compute::Tensor> srcTensors, dstTensors;
    std::unique_ptr<arm_compute::IFunction> ifunc;
};

class AclEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;

    [[nodiscard]] EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclEltwiseExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
