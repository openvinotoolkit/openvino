// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "nodes/executors/interpolate.hpp"
#include "nodes/executors/interpolate_config.hpp"

namespace ov::intel_cpu {

class RefInterpolateExecutor : public InterpolateExecutor {
public:
    explicit RefInterpolateExecutor(ExecutorContext::CPtr context) 
        : InterpolateExecutor(std::move(context)) {}
    
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

private:
    // Nearest neighbor interpolation
    void nearestInterpolation(const uint8_t* src_data,
                              uint8_t* dst_data,
                              const VectorDims& srcDims,
                              const VectorDims& dstDims);
    
    // Linear interpolation
    void linearInterpolation(const uint8_t* src_data,
                            uint8_t* dst_data,
                            const VectorDims& srcDims,
                            const VectorDims& dstDims);
    
    // Cubic interpolation
    void cubicInterpolation(const uint8_t* src_data,
                           uint8_t* dst_data,
                           const VectorDims& srcDims,
                           const VectorDims& dstDims);
    
    // Helper functions for coordinate transformation
    float getCoordinate(int index, float scale, InterpolateCoordTransMode mode);
    int getNearestIndex(float coord, int size, InterpolateNearestMode mode);
    
    // Interpolation tables for optimization
    void buildIndexTables();
    
    InterpolateAttrs attrs_;
    VectorDims srcDims_;
    VectorDims dstDims_;
    size_t dataTypeSize_;
    
    // Pre-computed index and weight tables
    std::vector<int> indexTable_;
    std::vector<float> weightTable_;
};

}  // namespace ov::intel_cpu