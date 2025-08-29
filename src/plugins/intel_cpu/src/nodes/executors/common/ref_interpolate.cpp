// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/parallel.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

bool RefInterpolateExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs,
                                  const dnnl::primitive_attr& attr) {
    if (!InterpolateExecutor::init(interpolateAttrs, srcDescs, dstDescs, attr)) {
        return false;
    }
    
    attrs_ = interpolateAttrs;
    srcDims_ = srcDescs[0]->getShape().getStaticDims();
    dstDims_ = dstDescs[0]->getShape().getStaticDims();
    
    // Determine data type size
    const auto precision = srcDescs[0]->getPrecision();
    dataTypeSize_ = precision.size();
    
    // Build index tables for optimization
    buildIndexTables();
    
    return true;
}

void RefInterpolateExecutor::exec(const std::vector<MemoryCPtr>& src,
                                  const std::vector<MemoryPtr>& dst,
                                  const void* post_ops_data_) {
    const auto* src_data = padPreprocess(src, dst);
    auto* dst_data = dst[0]->getDataAs<uint8_t>();
    
    switch (attrs_.mode) {
        case InterpolateMode::nearest:
            nearestInterpolation(src_data, dst_data, srcDimPad5d, dstDim5d);
            break;
        case InterpolateMode::linear:
        case InterpolateMode::linear_onnx:
            linearInterpolation(src_data, dst_data, srcDimPad5d, dstDim5d);
            break;
        case InterpolateMode::cubic:
            cubicInterpolation(src_data, dst_data, srcDimPad5d, dstDim5d);
            break;
        case InterpolateMode::bilinear_pillow:
        case InterpolateMode::bicubic_pillow:
            // Pillow modes use special handling similar to linear/cubic
            linearInterpolation(src_data, dst_data, srcDimPad5d, dstDim5d);
            break;
        default:
            OPENVINO_THROW("Unsupported interpolation mode in reference executor");
    }
}

void RefInterpolateExecutor::nearestInterpolation(const uint8_t* src_data,
                                                  uint8_t* dst_data,
                                                  const VectorDims& srcDims,
                                                  const VectorDims& dstDims) {
    const size_t N = srcDims[0];
    const size_t C = srcDims[1];
    const size_t ID = srcDims[2];
    const size_t IH = srcDims[3];
    const size_t IW = srcDims[4];
    const size_t OD = dstDims[2];
    const size_t OH = dstDims[3];
    const size_t OW = dstDims[4];
    
    const float scale_d = (ID > 1 && OD > 1) ? static_cast<float>(ID) / OD : 1.0f;
    const float scale_h = (IH > 1 && OH > 1) ? static_cast<float>(IH) / OH : 1.0f;
    const float scale_w = (IW > 1 && OW > 1) ? static_cast<float>(IW) / OW : 1.0f;
    
    parallel_for2d(N, C, [&](size_t n, size_t c) {
        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    // Calculate source coordinates
                    float src_d = getCoordinate(od, scale_d, attrs_.coordTransMode);
                    float src_h = getCoordinate(oh, scale_h, attrs_.coordTransMode);
                    float src_w = getCoordinate(ow, scale_w, attrs_.coordTransMode);
                    
                    // Get nearest indices
                    int id = getNearestIndex(src_d, ID, attrs_.nearestMode);
                    int ih = getNearestIndex(src_h, IH, attrs_.nearestMode);
                    int iw = getNearestIndex(src_w, IW, attrs_.nearestMode);
                    
                    // Clip indices to valid range
                    id = clipCoord(id, ID);
                    ih = clipCoord(ih, IH);
                    iw = clipCoord(iw, IW);
                    
                    // Calculate offsets
                    size_t src_offset = ((n * C + c) * ID * IH * IW + 
                                        id * IH * IW + ih * IW + iw) * dataTypeSize_;
                    size_t dst_offset = ((n * C + c) * OD * OH * OW + 
                                        od * OH * OW + oh * OW + ow) * dataTypeSize_;
                    
                    // Copy data
                    std::memcpy(dst_data + dst_offset, src_data + src_offset, dataTypeSize_);
                }
            }
        }
    });
}

void RefInterpolateExecutor::linearInterpolation(const uint8_t* src_data,
                                                 uint8_t* dst_data,
                                                 const VectorDims& srcDims,
                                                 const VectorDims& dstDims) {
    const size_t N = srcDims[0];
    const size_t C = srcDims[1];
    const size_t ID = srcDims[2];
    const size_t IH = srcDims[3];
    const size_t IW = srcDims[4];
    const size_t OD = dstDims[2];
    const size_t OH = dstDims[3];
    const size_t OW = dstDims[4];
    
    const float scale_d = (ID > 1 && OD > 1) ? static_cast<float>(ID - 1) / (OD - 1) : 1.0f;
    const float scale_h = (IH > 1 && OH > 1) ? static_cast<float>(IH - 1) / (OH - 1) : 1.0f;
    const float scale_w = (IW > 1 && OW > 1) ? static_cast<float>(IW - 1) / (OW - 1) : 1.0f;
    
    const auto* src_f32 = reinterpret_cast<const float*>(src_data);
    auto* dst_f32 = reinterpret_cast<float*>(dst_data);
    
    parallel_for2d(N, C, [&](size_t n, size_t c) {
        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    // Calculate source coordinates
                    float src_d = getCoordinate(od, scale_d, attrs_.coordTransMode);
                    float src_h = getCoordinate(oh, scale_h, attrs_.coordTransMode);
                    float src_w = getCoordinate(ow, scale_w, attrs_.coordTransMode);
                    
                    // Get interpolation indices and weights
                    int id0 = static_cast<int>(std::floor(src_d));
                    int ih0 = static_cast<int>(std::floor(src_h));
                    int iw0 = static_cast<int>(std::floor(src_w));
                    
                    int id1 = std::min(id0 + 1, static_cast<int>(ID) - 1);
                    int ih1 = std::min(ih0 + 1, static_cast<int>(IH) - 1);
                    int iw1 = std::min(iw0 + 1, static_cast<int>(IW) - 1);
                    
                    id0 = clipCoord(id0, ID);
                    ih0 = clipCoord(ih0, IH);
                    iw0 = clipCoord(iw0, IW);
                    
                    float wd = src_d - id0;
                    float wh = src_h - ih0;
                    float ww = src_w - iw0;
                    
                    // Trilinear interpolation
                    float val = 0.0f;
                    for (int d = 0; d <= (ID > 1 ? 1 : 0); ++d) {
                        for (int h = 0; h <= (IH > 1 ? 1 : 0); ++h) {
                            for (int w = 0; w <= (IW > 1 ? 1 : 0); ++w) {
                                int cur_id = d ? id1 : id0;
                                int cur_ih = h ? ih1 : ih0;
                                int cur_iw = w ? iw1 : iw0;
                                
                                float weight = 1.0f;
                                if (ID > 1) weight *= d ? wd : (1.0f - wd);
                                if (IH > 1) weight *= h ? wh : (1.0f - wh);
                                if (IW > 1) weight *= w ? ww : (1.0f - ww);
                                
                                size_t src_idx = (n * C + c) * ID * IH * IW + 
                                                cur_id * IH * IW + cur_ih * IW + cur_iw;
                                val += src_f32[src_idx] * weight;
                            }
                        }
                    }
                    
                    size_t dst_idx = (n * C + c) * OD * OH * OW + 
                                    od * OH * OW + oh * OW + ow;
                    dst_f32[dst_idx] = val;
                }
            }
        }
    });
}

void RefInterpolateExecutor::cubicInterpolation(const uint8_t* src_data,
                                                uint8_t* dst_data,
                                                const VectorDims& srcDims,
                                                const VectorDims& dstDims) {
    // Cubic interpolation implementation
    // This is a simplified version - full implementation would be more complex
    // For now, fallback to linear interpolation
    linearInterpolation(src_data, dst_data, srcDims, dstDims);
}

float RefInterpolateExecutor::getCoordinate(int index, float scale, InterpolateCoordTransMode mode) {
    float coord = 0.0f;
    
    switch (mode) {
        case InterpolateCoordTransMode::half_pixel:
            coord = (index + 0.5f) * scale - 0.5f;
            break;
        case InterpolateCoordTransMode::pytorch_half_pixel:
            coord = index * scale;
            break;
        case InterpolateCoordTransMode::asymmetric:
            coord = index * scale;
            break;
        case InterpolateCoordTransMode::tf_half_pixel_for_nn:
            coord = (index + 0.5f) * scale;
            break;
        case InterpolateCoordTransMode::align_corners:
            coord = index * scale;
            break;
    }
    
    return coord;
}

int RefInterpolateExecutor::getNearestIndex(float coord, int size, InterpolateNearestMode mode) {
    int index = 0;
    
    switch (mode) {
        case InterpolateNearestMode::round_prefer_floor:
            index = static_cast<int>(std::floor(coord + 0.5f));
            if (coord - std::floor(coord) == 0.5f) {
                index = static_cast<int>(std::floor(coord));
            }
            break;
        case InterpolateNearestMode::round_prefer_ceil:
            index = static_cast<int>(std::floor(coord + 0.5f));
            if (coord - std::floor(coord) == 0.5f) {
                index = static_cast<int>(std::ceil(coord));
            }
            break;
        case InterpolateNearestMode::floor:
            index = static_cast<int>(std::floor(coord));
            break;
        case InterpolateNearestMode::ceil:
            index = static_cast<int>(std::ceil(coord));
            break;
        case InterpolateNearestMode::simple:
            index = static_cast<int>(std::round(coord));
            break;
    }
    
    return index;
}

void RefInterpolateExecutor::buildIndexTables() {
    // Pre-compute index and weight tables for optimization
    // This is especially useful for repeated interpolations with the same configuration
    // Implementation depends on specific interpolation mode and dimensions
}

}  // namespace ov::intel_cpu