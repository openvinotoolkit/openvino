// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/interpolate.hpp"
#include "nodes/interpolate.h"
#include <memory>

namespace ov::intel_cpu {

class NewRefInterpolateExecutor : public InterpolateExecutor {
public:
    explicit NewRefInterpolateExecutor(ExecutorContext::CPtr context);
    
    bool init(const InterpolateAttrs& interpolateAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
              
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;
              
    [[nodiscard]] impl_desc_type getImplType() const override {
        return impl_desc_type::ref;
    }
    
    ~NewRefInterpolateExecutor() override = default;

private:
    // Use old reference executor internally with proper initialization
    std::shared_ptr<node::Interpolate::OldInterpolateRefExecutor> oldRefExecutor_;
    InterpolateAttrs attrs_;
    VectorDims srcDims_;
    VectorDims dstDims_;
    std::vector<float> dataScales_;
    std::vector<uint8_t> paddedSrcData_;
    bool hasPadding_ = false;
    
    void buildIndexWeightTables();
    void preprocessPadding(const std::vector<MemoryCPtr>& src);
};

}  // namespace ov::intel_cpu