// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate_ref.hpp"
#include "cpu_shape.h"
#include "utils/general_utils.h"
#include "openvino/core/parallel.hpp"
#include <cstring>
#include <iostream>

namespace ov::intel_cpu {

NewRefInterpolateExecutor::NewRefInterpolateExecutor(ExecutorContext::CPtr context) 
    : InterpolateExecutor(context) {
}

bool NewRefInterpolateExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs,
                                  const dnnl::primitive_attr& attr) {
    // Call base class init first
    if (!InterpolateExecutor::init(interpolateAttrs, srcDescs, dstDescs, attr)) {
        return false;
    }
    
    attrs_ = interpolateAttrs;
    
    // Extract dimensions
    if (!srcDescs.empty() && srcDescs[0]) {
        srcDims_ = srcDescs[0]->getShape().getStaticDims();
    }
    if (!dstDescs.empty() && dstDescs[0]) {
        dstDims_ = dstDescs[0]->getShape().getStaticDims();
    }
    
    // Check if we have valid dimensions
    if (srcDims_.empty() || dstDims_.empty()) {
        return false;
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
    
    // Note: Defer executor creation until runtime due to shape mismatches between init and runtime
    // The init-time srcDims_ may be different from runtime source dimensions
    // buildIndexWeightTables();
    
    return true;
}

void NewRefInterpolateExecutor::buildIndexWeightTables() {
    // std::cerr << "[DEBUG BuildTables] srcDims_: [";
    // Debug output removed
    // for (size_t i = 0; i < srcDims_.size(); ++i) {
    //     std::cerr << srcDims_[i] << (i < srcDims_.size()-1 ? "," : "");
    // }
    // std::cerr << "], dstDims_: [";
    // for (size_t i = 0; i < dstDims_.size(); ++i) {
    //     std::cerr << dstDims_[i] << (i < dstDims_.size()-1 ? "," : "");
    // }
    // std::cerr << "]" << std::endl;
    
    // Create old reference executor with init-time dimensions and scales
    // This ensures consistency with what the original node expected
    if (srcDims_.empty() || dstDims_.empty() || dataScales_.empty()) {
        return;
    }
    
    try {
        // Use init-time dimensions that match the original dataScales_ expectations
        oldRefExecutor_ = std::make_shared<node::Interpolate::OldInterpolateRefExecutor>(
            attrs_,
            srcDims_,      // Use init-time source dimensions 
            dstDims_,      // Use init-time destination dimensions
            dataScales_);  // Use original dataScales_ from init
            
        if (!oldRefExecutor_) {
            // std::cerr << "[DEBUG BuildTables] Failed to create old reference executor - nullptr" << std::endl;
            return;
        }
        
        // Debug output removed for production
        // std::cerr << "[DEBUG BuildTables] Created old executor with srcDims_: [..." << std::endl;
        
    } catch (const std::exception& e) {
        // std::cerr << "[DEBUG BuildTables] Exception creating old reference executor: " << e.what() << std::endl;
        oldRefExecutor_ = nullptr;
    }
}

void NewRefInterpolateExecutor::preprocessPadding(const std::vector<MemoryCPtr>& src) {
    if (!hasPadding_ || src.empty()) {
        return;
    }
    
    const auto srcMemory = src[0];
    const auto& srcDim = srcDims_;
    const auto srcDimRuntime = srcMemory->getStaticDims();
    
    // Debug output removed for production
    // std::cerr << "[DEBUG Padding] dimensions logged" << std::endl;
    
    // Use runtime dimensions instead of init-time dimensions
    const auto srcDimPadded = getPaddedInputShape(srcDimRuntime, attrs_.padBegin, attrs_.padEnd);
    
    const auto srcDim5d = to5Dim(srcDimRuntime);
    const auto srcDimPad5d = to5Dim(srcDimPadded);
    const auto srcDataSize = srcMemory->getDesc().getPrecision().size();
    
    int dimSize = srcDimRuntime.size();
    int padB0 = (dimSize > 2) ? attrs_.padBegin[0] : 0;
    int padB1 = (dimSize > 2) ? attrs_.padBegin[1] : 0;
    int padB2 = (dimSize == 5) ? attrs_.padBegin[dimSize - 3] : 0;
    int padB3 = (dimSize > 1) ? attrs_.padBegin[dimSize - 2] : 0;
    int padB4 = attrs_.padBegin[dimSize - 1];
    
    auto getBlockND = [](const VectorDims& shape) -> VectorDims {
        // Match the old implementation exactly: return shapeRank + 1 elements
        int shapeRank = shape.size();
        VectorDims blockND(shapeRank + 1, 1);
        for (int i = shapeRank - 1; i >= 0; i--) {
            blockND[i] = shape[i] * blockND[i + 1];
        }
        return blockND;
    };
    
    VectorDims inShapeBlock = getBlockND(srcDim5d);
    VectorDims inShapePadBlock = getBlockND(srcDimPad5d);
    
    const uint8_t* src_data_origin = srcMemory->getDataAs<const uint8_t>();
    
    // std::cerr << "[DEBUG Padding] Layout type: " << static_cast<int>(attrs_.layout) 
    //           << " (0=planar, 1=by_channel, 2=block)" << std::endl;
    // std::cerr << "[DEBUG Padding] srcDim5d: [" << srcDim5d[0] << "," << srcDim5d[1] << "," << srcDim5d[2] << "," << srcDim5d[3] << "," << srcDim5d[4] << "]" << std::endl;
    // std::cerr << "[DEBUG Padding] srcDimPad5d: [" << srcDimPad5d[0] << "," << srcDimPad5d[1] << "," << srcDimPad5d[2] << "," << srcDimPad5d[3] << "," << srcDimPad5d[4] << "]" << std::endl;
    // std::cerr << "[DEBUG Padding] inShapePadBlock[0]: " << inShapePadBlock[0] << ", srcDataSize: " << srcDataSize << std::endl;
    
    if (attrs_.layout == InterpolateLayoutType::planar) {
        paddedSrcData_.resize(inShapePadBlock[0] * srcDataSize, 0);
        uint8_t* src_data_pad = paddedSrcData_.data();
        
        // std::cerr << "[DEBUG Padding] About to copy planar data" << std::endl;
        // std::cerr << "[DEBUG Padding] Source 5D dims: [" << srcDim5d[0] << "," << srcDim5d[1] << "," << srcDim5d[2] << "," << srcDim5d[3] << "," << srcDim5d[4] << "]" << std::endl;
        // std::cerr << "[DEBUG Padding] Padded 5D dims: [" << srcDimPad5d[0] << "," << srcDimPad5d[1] << "," << srcDimPad5d[2] << "," << srcDimPad5d[3] << "," << srcDimPad5d[4] << "]" << std::endl;
        // std::cerr << "[DEBUG Padding] Padding offsets: [" << padB0 << "," << padB1 << "," << padB2 << "," << padB3 << "," << padB4 << "]" << std::endl;
        
        // Debug output removed for production
        // Test first few source values (debug code removed)
        
        // Debug first iteration
        bool debugged = false;
        ov::parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], 
            [&](int n, int c, int d, int h) {
                const uint8_t* src_ptr = src_data_origin + 
                    (inShapeBlock[1] * n + inShapeBlock[2] * c +
                     inShapeBlock[3] * d + inShapeBlock[4] * h) * srcDataSize;
                uint8_t* dst_ptr = src_data_pad + 
                    (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                     inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) * srcDataSize;
                
                if (!debugged && n == 0 && c == 0 && d == 0 && h == 0) {
                    size_t src_offset = (inShapeBlock[1] * n + inShapeBlock[2] * c + inShapeBlock[3] * d + inShapeBlock[4] * h);
                    size_t dst_offset = (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) + inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4);
                    // std::cerr << "[DEBUG Copy] First copy: n=" << n << " c=" << c << " d=" << d << " h=" << h 
                    //           << ", src_offset=" << src_offset << " dst_offset=" << dst_offset 
                    //           << ", copy_size=" << srcDim5d[4] << " elements" << std::endl;
                    // std::cerr << "[DEBUG Copy] inShapeBlock: [" << inShapeBlock[0] << "," << inShapeBlock[1] << "," << inShapeBlock[2] << "," << inShapeBlock[3] << "," << inShapeBlock[4] << "]" << std::endl;
                    // std::cerr << "[DEBUG Copy] inShapePadBlock: [" << inShapePadBlock[0] << "," << inShapePadBlock[1] << "," << inShapePadBlock[2] << "," << inShapePadBlock[3] << "," << inShapePadBlock[4] << "]" << std::endl;
                    debugged = true;
                }
                
                std::memcpy(dst_ptr, src_ptr, srcDim5d[4] * srcDataSize);
            });
            
        // Debug output removed for production
        // Test first few padded values (debug code removed)
    } else if (attrs_.layout == InterpolateLayoutType::by_channel) {
        paddedSrcData_.resize(inShapePadBlock[0] * srcDataSize, 0);
        uint8_t* src_data_pad = paddedSrcData_.data();
        
        ov::parallel_for4d(srcDim5d[0], srcDim5d[2], srcDim5d[3], srcDim5d[4],
            [&](int n, int d, int h, int w) {
                const uint8_t* src_ptr = src_data_origin +
                    (inShapeBlock[1] * n + inShapeBlock[3] * d + 
                     inShapeBlock[4] * h + w) * srcDim5d[1] * srcDataSize;
                uint8_t* dst_ptr = src_data_pad +
                    (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[3] * (d + padB2) +
                     inShapePadBlock[4] * (h + padB3) + (w + padB4)) * srcDim5d[1] * srcDataSize;
                std::memcpy(dst_ptr, src_ptr, srcDim5d[1] * srcDataSize);
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

const uint8_t* NewRefInterpolateExecutor::padPreprocess(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    if (!hasPadding_ || src.empty()) {
        return nullptr;
    }
    
    // Call the preprocessing function
    preprocessPadding(src);
    
    // Return pointer to padded data
    if (paddedSrcData_.empty()) {
        // std::cerr << "[DEBUG padPreprocess] paddedSrcData_ is empty, returning nullptr" << std::endl;
        return nullptr;
    }
    
    // std::cerr << "[DEBUG padPreprocess] Returning padded data ptr, size: " << paddedSrcData_.size() << std::endl;
    return paddedSrcData_.data();
}

void NewRefInterpolateExecutor::exec(const std::vector<MemoryCPtr>& src,
                                  const std::vector<MemoryPtr>& dst,
                                  const void* post_ops_data_) {
    // // std::cerr << "[DEBUG NewRef] exec called with src.size()=" << src.size() << " dst.size()=" << dst.size() << std::endl;
    
    if (!oldRefExecutor_) {
        // std::cerr << "[DEBUG NewRef] Old ref executor is null, creating with runtime dimensions" << std::endl;
        
        // Get runtime dimensions
        auto srcDimRuntime = src[0]->getStaticDims();
        auto dstDimRuntime = dst[0]->getStaticDims();
        
        // For padding cases: build executor to expect padded source dimensions with no scaling
        // For no-padding cases: build executor for direct source-to-destination scaling  
        VectorDims executorSrcDims = srcDimRuntime;
        VectorDims executorDstDims = dstDimRuntime;
        std::vector<float> executorScales;
        
        // Calculate scales exactly like the old implementation
        executorScales.reserve(executorSrcDims.size());
        for (size_t i = 0; i < executorSrcDims.size(); ++i) {
            if (hasPadding_ && i < attrs_.padBegin.size() && i < attrs_.padEnd.size()) {
                // For padded data, use padded source dimensions 
                size_t srcDimWithPadding = executorSrcDims[i] + attrs_.padBegin[i] + attrs_.padEnd[i];
                executorScales.push_back(static_cast<float>(executorDstDims[i]) / static_cast<float>(srcDimWithPadding));
            } else {
                // For non-padded data, use original dimensions
                executorScales.push_back(static_cast<float>(executorDstDims[i]) / static_cast<float>(executorSrcDims[i]));
            }
        }
        // Debug output removed for production
        // std::cerr << "[DEBUG NewRef] Building old executor" << std::endl;
        // std::cerr << "[DEBUG NewRef] attrs_.mode=" << static_cast<int>(attrs_.mode) 
        //           << ", attrs_.coordTransMode=" << static_cast<int>(attrs_.coordTransMode) 
        //           << ", attrs_.layout=" << static_cast<int>(attrs_.layout) 
        //           << ", attrs_.nearestMode=" << static_cast<int>(attrs_.nearestMode) 
        //           << ", attrs_.antialias=" << attrs_.antialias
        //           << ", attrs_.cubeCoeff=" << attrs_.cubeCoeff
        //           << ", attrs_.shapeCalcMode=" << static_cast<int>(attrs_.shapeCalcMode)
        //           << ", attrs_.inPrc=" << attrs_.inPrc.get_type_name() 
        //           << ", attrs_.outPrc=" << attrs_.outPrc.get_type_name() << std::endl;
        
        try {
            oldRefExecutor_ = std::make_shared<node::Interpolate::OldInterpolateRefExecutor>(
                attrs_,
                executorSrcDims, 
                executorDstDims,  
                executorScales);
                
            if (!oldRefExecutor_) {
                // std::cerr << "[DEBUG NewRef] Failed to create old reference executor" << std::endl;
                return;
            }
        } catch (const std::exception& e) {
            // std::cerr << "[DEBUG NewRef] Exception creating old executor: " << e.what() << std::endl;
            return;
        }
    }
    
    if (!oldRefExecutor_ || src.empty() || dst.empty()) {
        // std::cerr << "[DEBUG NewRef] Cannot execute: oldRefExecutor=" << (oldRefExecutor_ ? "valid" : "null") 
        //           << " src.empty=" << src.empty() << " dst.empty=" << dst.empty() << std::endl;
        return;
    }
    
    const uint8_t* src_data = nullptr;
    
    // Validate source memory
    if (src[0]) {
        auto srcDims = src[0]->getStaticDims();
        // Debug output removed for production
        // std::cerr << "[DEBUG NewRef] Source info logged" << std::endl;
    }
    
    // Match old implementation exactly: prepare padded data if needed
    if (hasPadding_) {
        // std::cerr << "[DEBUG NewRef] Has padding, preprocessing data before passing to old executor" << std::endl;
        preprocessPadding(src);
        src_data = paddedSrcData_.data();
        
        // Debug output removed for production
        // Debug: print first few padded source data values (removed)
    } else {
        // No padding, use original data
        src_data = src[0]->getDataAs<const uint8_t>();
        // std::cerr << "[DEBUG NewRef] No padding, using original source data" << std::endl;
        
        // Debug output removed for production
        // Debug: print first few original source data values (removed)
    }
    
    uint8_t* dst_data = dst[0]->getDataAs<uint8_t>();
    
    // Validate destination memory
    if (dst[0]) {
        auto dstDims = dst[0]->getStaticDims();
        // Debug output removed for production
        // std::cerr << "[DEBUG NewRef] Destination info logged" << std::endl;
    }
    
    // std::cerr << "[DEBUG NewRef] About to call oldRefExecutor_->exec with src_data=" << (void*)src_data << " dst_data=" << (void*)dst_data << std::endl;
    
    // Execute using old reference executor
    oldRefExecutor_->exec(src_data, dst_data, post_ops_data_);
    
    // std::cerr << "[DEBUG NewRef] oldRefExecutor_->exec completed" << std::endl;
}

}  // namespace ov::intel_cpu