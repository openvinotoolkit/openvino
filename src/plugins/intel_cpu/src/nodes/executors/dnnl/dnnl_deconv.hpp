// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/deconv.hpp"
#include "utils/debug_capabilities.h"
#include "nodes/common/dnnl_executor.h"

namespace ov {
namespace intel_cpu {

class DNNLDeconvExecutor : public DeconvExecutor {
public:
    DNNLDeconvExecutor();
    using AttrPtr = std::shared_ptr<dnnl::primitive_attr>;

    bool init(const DeconvAttrs& deconvAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;
    impl_desc_type getImplType() const override { return implType; }

private:
    DeconvAttrs deconvAttrs;
    impl_desc_type implType = impl_desc_type::any;
    std::shared_ptr<DnnlExecutor> dnnlExecPtr = nullptr;
};

class DNNLDeconvExecutorBuilder : public DeconvExecutorBuilder {
public:
    bool isSupported(const DeconvAttrs& deconvAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    DeconvExecutorPtr makeExecutor() const override {
        return std::make_shared<DNNLDeconvExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov