// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

#define MAX_INPUT_INTERPOLATE 8

namespace ov {
namespace intel_cpu {

enum InterpolateLayoutType {
    planar,
    block,
    by_channel
};

enum InterpolateMode {
    nearest,
    linear,
    linear_onnx,
    cubic,
    bilinear_pillow,
    bicubic_pillow
};

enum InterpolateCoordTransMode {
    half_pixel,
    pytorch_half_pixel,
    asymmetric,
    tf_half_pixel_for_nn,
    align_corners
};

enum class InterpolateNearestMode {
    round_prefer_floor,
    round_prefer_ceil,
    floor,
    ceil,
    simple
};

enum class InterpolateShapeCalcMode {
    sizes,
    scales
};

struct InterpolateAttrs {
    InterpolateShapeCalcMode shapeCalcMode = InterpolateShapeCalcMode::sizes;
    InterpolateMode mode = InterpolateMode::nearest;
    InterpolateCoordTransMode coordTransMode = InterpolateCoordTransMode::half_pixel;
    InterpolateNearestMode nearestMode = InterpolateNearestMode::round_prefer_floor;
    bool antialias = false;
    float cubeCoeff = -0.75;
    std::vector<int> padBegin;
    std::vector<int> padEnd;
    ov::element::Type inPrc;
    ov::element::Type outPrc;
    InterpolateLayoutType layout;
    std::vector<float> dataScales;
    bool hasPad = false;
};

inline VectorDims getPaddedInputShape(const VectorDims &srcDims,
                                const std::vector<int> &padBegin,
                                const std::vector<int> &padEnd) {
    VectorDims paddedShape;
    int dataRank = srcDims.size();
    for (int i = 0; i < dataRank; i++) {
        paddedShape.push_back(srcDims[i] + padBegin[i] + padEnd[i]);
    }
    return paddedShape;
}

inline int clipCoord(int pos, int length) {
    return std::max(static_cast<int>(0), std::min(pos, length - 1));
}

inline size_t getSpatialDimsNum(const Dim rank) {
    switch (rank) {
        case 1:
        case 3:
            return 1;
        case 2:
        case 4:
            return 2;
        case 5:
            return 3;
        default:
            OPENVINO_THROW("Can't define number spatial");
    }
}

// w/hw/ncw/nchw/ncdhw to ncdhw
inline VectorDims to5Dim(VectorDims casesDim) {
    size_t caseSize = casesDim.size();
    VectorDims dim5(5, 1lu);
    dim5[4] = casesDim[caseSize - 1];
    if (caseSize > 1) {
        dim5[3] = casesDim[caseSize - 2];
    }
    if (caseSize > 2) {
        dim5[0] = casesDim[0];
    }
    if (caseSize > 3) {
        dim5[1] = casesDim[1];
    }
    if (caseSize > 4) {
        dim5[2] = casesDim[2];
    }
    if (caseSize == 3) {  // nhw -> ncw
        dim5[1] = dim5[3];
        dim5[3] = 1lu;
    }
    return dim5;
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0f, 1 - std::abs(x));
}

class InterpolateExecutor {
public:
    static constexpr size_t DATA_ID = 0;
    static constexpr size_t TARGET_SHAPE_ID = 1;
    static constexpr size_t SCALES_ID = 2;
    static constexpr size_t AXES_ID = 3;
    static constexpr int CUBIC_GRID_LEN = 4;
    InterpolateExecutor(const ExecutorContext::CPtr context) : _context(context) {}

    virtual bool init(const InterpolateAttrs& interpolateAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr);
    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) = 0;
    virtual impl_desc_type getImplType() const = 0;

    virtual ~InterpolateExecutor() = default;
    VectorDims getSrcDimPad5d() const { return srcDimPad5d; }
    const uint8_t* padPreprocess(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst);

private:
    void buildTblNN(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales,
                    InterpolateLayoutType layout, InterpolateNearestMode nearestMode);
    void buildTblLinearOnnx(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales,
                            InterpolateLayoutType layout);
    void buildTblLinear(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales, int kernel_width,
                        bool antialias);
    void buildTblCubic(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales, float cubicCoeff,
                       InterpolateLayoutType layout);

    float coordTransToInput(int outCoord, float scale, int inShape, int outShape) const;
    int nearestRound(float origin, bool isDownsample, InterpolateNearestMode nearestMode) const;
    void linearOnnxCF(int outCoord, float scale, int inShape, int outShape, int& index0, int& index1, float& weight0, float& weight1);
    std::vector<float> getCubicCoeffs(float mantissa, float a);

protected:
    InterpolateAttrs interpAttrs;
    VectorDims srcDimPad5d, dstDim5d;
    size_t srcDataSize, dstDataSize;
    int spatialDimSize;
    size_t dataRank;
    std::vector<int> indexTable;
    const ExecutorContext::CPtr _context;
};

using InterpolateExecutorPtr = std::shared_ptr<InterpolateExecutor>;
using InterpolateExecutorCPtr = std::shared_ptr<const InterpolateExecutor>;

class InterpolateExecutorBuilder {
public:
    ~InterpolateExecutorBuilder() = default;
    virtual bool isSupported(const InterpolateAttrs& InterpolateAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual InterpolateExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using InterpolateExecutorBuilderPtr = std::shared_ptr<InterpolateExecutorBuilder>;
using InterpolateExecutorBuilderCPtr = std::shared_ptr<const InterpolateExecutorBuilder>;
} // namespace intel_cpu
} // namespace ov