// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/interpolate.hpp"
#include "nodes/interpolate.h"
#include <memory>
#include <vector>

namespace ov::intel_cpu {

class JitInterpolateExecutor : public InterpolateExecutor {
public:
    explicit JitInterpolateExecutor(ExecutorContext::CPtr context);
    
    bool init(const InterpolateAttrs& interpolateAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
              
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;
              
    [[nodiscard]] impl_desc_type getImplType() const override;  // Moved to cpp for ISA detection
    
    bool update(const MemoryArgs& memory) override;
    
    ~JitInterpolateExecutor() override = default;

private:
    // Configuration
    InterpolateAttrs attrs_;
    VectorDims srcDims_;
    VectorDims dstDims_;
    VectorDims srcDimsPadded_;
    std::vector<float> dataScales_;
    bool hasPadding_ = false;
    
    // Precision
    ov::element::Type dataPrecision_;
    
    // Use old JIT executor internally
    std::shared_ptr<node::Interpolate::OldInterpolateJitExecutor> oldJitExecutor_;
};

}  // namespace ov::intel_cpu