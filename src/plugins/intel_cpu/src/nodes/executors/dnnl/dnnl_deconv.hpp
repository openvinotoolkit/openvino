// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/deconv.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

using DefaultDeconvDescs = std::pair<dnnl::convolution_backward_data::primitive_desc,
        dnnl::convolution_forward::primitive_desc>;


DefaultDeconvDescs createDescriptorInternalDefault(const dnnl::memory::desc& in_candidate,
                                                   const dnnl::memory::desc& wgh_candidate,
                                                   const dnnl::memory::desc& out_candidate,
                                                   const dnnl::algorithm alg,
                                                   const std::vector<ptrdiff_t>& stride,
                                                   const std::vector<ptrdiff_t>& dilation,
                                                   const ov::CoordinateDiff& paddingL,
                                                   const ov::CoordinateDiff& paddingR,
                                                   const dnnl::primitive_attr& attr,
                                                   const dnnl::engine& engine);

dnnl::primitive_desc createDescriptorInternalInt8(const dnnl::memory::desc& in_candidate,
                                                  const dnnl::memory::desc& wgh_candidate,
                                                  const dnnl::memory::desc& bias_candidate,
                                                  const dnnl::memory::desc& out_candidate,
                                                  const bool with_bias,
                                                  const std::vector<ptrdiff_t>& stride,
                                                  const std::vector<ptrdiff_t>& dilation,
                                                  const ov::CoordinateDiff& paddingL,
                                                  const ov::CoordinateDiff& paddingR,
                                                  const dnnl::primitive_attr& attr,
                                                  const dnnl::engine& engine);

DefaultDeconvDescs createDefaultMkldnnDeconvDesc(const dnnl::memory::desc& srcDesc,
                                                 const dnnl::memory::desc& wghDesc,
                                                 const dnnl::memory::desc& dstDesc,
                                                 bool isWinograd,
                                                 const std::vector<ptrdiff_t>& stride,
                                                 const std::vector<ptrdiff_t>& dilation,
                                                 const ov::CoordinateDiff& paddingL,
                                                 const ov::CoordinateDiff& paddingR,
                                                 const dnnl::primitive_attr& attr,
                                                 const dnnl::engine& engine);

dnnl::primitive_desc createInt8MkldnnDeconvDesc(const dnnl::memory::desc& srcDesc,
                                                const dnnl::memory::desc& wghDesc,
                                                const dnnl::memory::desc& biasDesc,
                                                const dnnl::memory::desc& dstDesc,
                                                const bool withBias,
                                                const std::vector<ptrdiff_t>& stride,
                                                const std::vector<ptrdiff_t>& dilation,
                                                const ov::CoordinateDiff& paddingL,
                                                const ov::CoordinateDiff& paddingR,
                                                const dnnl::primitive_attr& attr,
                                                const dnnl::engine& engine);

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
              const void *post_ops_data_,
              const dnnl::stream &strm) override;
    impl_desc_type getImplType() const override { return dnnlExecPtr->getImplementationType(); }

private:
    DeconvAttrs dnnlDeconvAttrs;
    std::shared_ptr<std::unordered_map<int, dnnl::memory>> primArgsPtr;
    std::shared_ptr<DnnlExecutor> dnnlExecPtr = nullptr;

    class DeconvExecutorDefault : public DnnlExecutor {
    public:
        DeconvExecutorDefault(const dnnl::convolution_backward_data::primitive_desc& pd,
                              const dnnl::memory::desc& inMemDesc,
                              const dnnl::memory::desc& weightMemDesc,
                              const dnnl::memory::desc& outMemDesc,
                              const dnnl::engine& engine);
    };

    class DeconvExecutorInt8 : public DnnlExecutor {
    public:
        DeconvExecutorInt8(const dnnl::deconvolution_forward::primitive_desc& pd,
                           const dnnl::memory::desc& inMemDesc,
                           const dnnl::memory::desc& weightMemDesc,
                           const dnnl::memory::desc& outMemDesc,
                           const dnnl::engine& engine);
    };
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