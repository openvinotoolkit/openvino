// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_utils.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "nodes/executors/deconv.hpp"
#include "src/cpu/CpuTypes.h"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

struct ACLDeconvTensorInfo {
    arm_compute::TensorInfo srcTensorInfo;
    arm_compute::TensorInfo weiTensorInfo;
    arm_compute::TensorInfo biasTensorInfo;
    arm_compute::TensorInfo dstTensorInfo;
    arm_compute::PadStrideInfo deconv_info;
};

ACLDeconvTensorInfo getACLDeconvTensorInfo(const DeconvAttrs& deconvAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs);

class AclDeconvExecutor : public DeconvExecutor {
public:
    explicit AclDeconvExecutor(const ExecutorContext::CPtr context);
    bool init(const DeconvAttrs& deconvAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override {
        return implType;
    }

private:
    DeconvAttrs deconvAttrs;
    impl_desc_type implType = impl_desc_type::gemm_acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor weiTensor;
    arm_compute::Tensor biasTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEDeconvolutionLayer> deconv = nullptr;
};

class AclDeconvExecutorBuilder : public DeconvExecutorBuilder {
public:
    static bool customIsSupported(const DeconvAttrs& deconvAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs);

    [[nodiscard]] bool isSupported(const DeconvAttrs& deconvAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override {
        return customIsSupported(deconvAttrs, srcDescs, dstDescs);
    }

    [[nodiscard]] DeconvExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclDeconvExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
