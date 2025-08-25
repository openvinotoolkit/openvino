// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate_jit.hpp"
#include "cpu_shape.h"
#include "utils/general_utils.h"
#include <cpu/x64/cpu_isa_traits.hpp>

namespace ov::intel_cpu {

using namespace dnnl::impl::cpu;

JitInterpolateExecutor::JitInterpolateExecutor(ExecutorContext::CPtr context) 
    : InterpolateExecutor(context) {
}

bool JitInterpolateExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs,
                                  const dnnl::primitive_attr& attr) {
    // Call base class init first
    if (!InterpolateExecutor::init(interpolateAttrs, srcDescs, dstDescs, attr)) {
        return false;
    }
    
    attrs_ = interpolateAttrs;
    
    // Extract dimensions
    if (!srcDescs.empty()) {
        srcDims_ = srcDescs[0]->getShape().getStaticDims();
    }
    if (!dstDescs.empty()) {
        dstDims_ = dstDescs[0]->getShape().getStaticDims();
    }
    
    // Check if we have padding
    hasPadding_ = false;
    for (size_t i = 0; i < attrs_.padBegin.size(); i++) {
        if (attrs_.padBegin[i] != 0 || attrs_.padEnd[i] != 0) {
            hasPadding_ = true;
            break;
        }
    }
    
    // Calculate or use provided scales
    dataScales_ = attrs_.dataScales;
    if (dataScales_.empty() && !srcDims_.empty() && !dstDims_.empty()) {
        dataScales_.reserve(srcDims_.size());
        for (size_t i = 0; i < srcDims_.size(); i++) {
            if (hasPadding_) {
                // Account for padding in scale calculation
                size_t srcDimPadded = srcDims_[i];
                if (i >= srcDims_.size() - attrs_.padBegin.size()) {
                    size_t padIdx = i - (srcDims_.size() - attrs_.padBegin.size());
                    srcDimPadded += attrs_.padBegin[padIdx] + attrs_.padEnd[padIdx];
                }
                dataScales_.push_back(static_cast<float>(dstDims_[i]) / static_cast<float>(srcDimPadded));
            } else {
                dataScales_.push_back(static_cast<float>(dstDims_[i]) / static_cast<float>(srcDims_[i]));
            }
        }
    }
    
    // Build index and weight tables for interpolation
    buildIndexWeightTables();
    
    return true;
}

void JitInterpolateExecutor::buildIndexWeightTables() {
    // Determine padded source dimensions
    VectorDims srcDimsPadded = srcDims_;
    if (hasPadding_) {
        srcDimsPadded = getPaddedInputShape(srcDims_, attrs_.padBegin, attrs_.padEnd);
    }
    
    // Create old JIT executor with padded dimensions
    // The old executor builds the index/weight tables in its constructor
    try {
        oldJitExecutor_ = std::make_shared<node::Interpolate::OldInterpolateJitExecutor>(
            attrs_,
            srcDimsPadded,  // Use padded dimensions
            dstDims_,
            dataScales_,
            dnnl::primitive_attr());
    } catch (const std::exception& e) {
        // Failed to create JIT executor
        oldJitExecutor_ = nullptr;
    }
}

void JitInterpolateExecutor::preprocessPadding(const std::vector<MemoryCPtr>& src) {
    if (!hasPadding_ || src.empty()) {
        return;
    }
    
    const auto srcMemory = src[0];
    const auto& srcDim = srcDims_;
    const auto srcDimPadded = getPaddedInputShape(srcDims_, attrs_.padBegin, attrs_.padEnd);
    
    const auto srcDim5d = to5Dim(srcDim);
    const auto srcDimPad5d = to5Dim(srcDimPadded);
    const auto srcDataSize = srcMemory->getDesc()->getPrecision().size();
    
    int dimSize = srcDim.size();
    int padB0 = (dimSize > 2) ? attrs_.padBegin[0] : 0;
    int padB1 = (dimSize > 2) ? attrs_.padBegin[1] : 0;
    int padB2 = (dimSize == 5) ? attrs_.padBegin[dimSize - 3] : 0;
    int padB3 = (dimSize > 1) ? attrs_.padBegin[dimSize - 2] : 0;
    int padB4 = attrs_.padBegin[dimSize - 1];
    
    auto getBlockND = [](const VectorDims& shape) -> VectorDims {
        VectorDims blocks(shape.size());
        blocks[shape.size() - 1] = 1;
        for (int i = shape.size() - 2; i >= 0; i--) {
            blocks[i] = blocks[i + 1] * shape[i + 1];
        }
        return blocks;
    };
    
    VectorDims inShapeBlock = getBlockND(srcDim5d);
    VectorDims inShapePadBlock = getBlockND(srcDimPad5d);
    
    const uint8_t* src_data_origin = srcMemory->getDataAs<const uint8_t>();
    
    if (attrs_.layout == InterpolateLayoutType::planar) {
        paddedSrcData_.resize(inShapePadBlock[0] * srcDataSize, 0);
        uint8_t* src_data_pad = paddedSrcData_.data();
        
        parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], 
            [&](int n, int c, int d, int h) {
                const uint8_t* src_ptr = src_data_origin + 
                    (inShapeBlock[1] * n + inShapeBlock[2] * c +
                     inShapeBlock[3] * d + inShapeBlock[4] * h) * srcDataSize;
                uint8_t* dst_ptr = src_data_pad + 
                    (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                     inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) * srcDataSize;
                cpu_memcpy(dst_ptr, src_ptr, srcDim5d[4] * srcDataSize);
            });
    } else if (attrs_.layout == InterpolateLayoutType::by_channel) {
        paddedSrcData_.resize(inShapePadBlock[0] * srcDataSize, 0);
        uint8_t* src_data_pad = paddedSrcData_.data();
        
        parallel_for4d(srcDim5d[0], srcDim5d[2], srcDim5d[3], srcDim5d[4],
            [&](int n, int d, int h, int w) {
                const uint8_t* src_ptr = src_data_origin +
                    (inShapeBlock[1] * n + inShapeBlock[3] * d + 
                     inShapeBlock[4] * h + w) * srcDim5d[1] * srcDataSize;
                uint8_t* dst_ptr = src_data_pad +
                    (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[3] * (d + padB2) +
                     inShapePadBlock[4] * (h + padB3) + (w + padB4)) * srcDim5d[1] * srcDataSize;
                cpu_memcpy(dst_ptr, src_ptr, srcDim5d[1] * srcDataSize);
            });
    } else {
        // Block layout - simplified padding
        size_t totalSize = 1;
        for (auto& dim : srcDimPadded) {
            totalSize *= dim;
        }
        paddedSrcData_.resize(totalSize * srcDataSize, 0);
        // For block layout, we'll let the old executor handle it
    }
}

void JitInterpolateExecutor::exec(const std::vector<MemoryCPtr>& src,
                                  const std::vector<MemoryPtr>& dst,
                                  const void* post_ops_data_) {
    if (!oldJitExecutor_ || src.empty() || dst.empty()) {
        return;
    }
    
    const uint8_t* src_data = nullptr;
    
    if (hasPadding_) {
        // Preprocess padding
        preprocessPadding(src);
        src_data = paddedSrcData_.data();
    } else {
        // Use original source data
        src_data = src[0]->getDataAs<const uint8_t>();
    }
    
    uint8_t* dst_data = dst[0]->getDataAs<uint8_t>();
    
    // Execute using old JIT executor
    oldJitExecutor_->exec(src_data, dst_data, post_ops_data_);
}

}  // namespace ov::intel_cpu