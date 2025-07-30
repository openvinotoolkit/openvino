// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

static constexpr int MAX_INPUT_INTERPOLATE = 8;

namespace ov::intel_cpu {

enum InterpolateLayoutType : uint8_t { planar, block, by_channel };

enum InterpolateMode : uint8_t { nearest, linear, linear_onnx, cubic, bilinear_pillow, bicubic_pillow };

enum InterpolateCoordTransMode : uint8_t {
    half_pixel,
    pytorch_half_pixel,
    asymmetric,
    tf_half_pixel_for_nn,
    align_corners
};

enum class InterpolateNearestMode : uint8_t { round_prefer_floor, round_prefer_ceil, floor, ceil, simple };

enum class InterpolateShapeCalcMode : uint8_t { sizes, scales };

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
    InterpolateLayoutType layout = InterpolateLayoutType::planar;
    std::vector<float> dataScales;
    bool hasPad = false;
    // Some FEs or preprocessing step resize spatial dimension for tensors with NHWC layout memory,
    // but import them with a planar layout[abcd] with axis[1,2] for convenience. In this case, for pillow modes without
    // pad, the nhwc layout path and the specific kernel(nhwc layout executor) can be used for this planar layout and
    // axis settings(NCHWAsNHWC is true) to get better perf. To this end the following mapping is used:
    // 1. logical shape alignment [abcd-nhwc] to [adbc-nchw].
    // 2. axis alignment [1,2] to [2,3].
    // 3. config planar layout support and treated it as channel_first layout.
    bool NCHWAsNHWC = false;
};

inline VectorDims getPaddedInputShape(const VectorDims& srcDims,
                                      const std::vector<int>& padBegin,
                                      const std::vector<int>& padEnd) {
    VectorDims paddedShape;
    int dataRank = srcDims.size();
    for (int i = 0; i < dataRank; i++) {
        paddedShape.push_back(srcDims[i] + padBegin[i] + padEnd[i]);
    }
    return paddedShape;
}

inline int clipCoord(int pos, int length) {
    return std::max(0, std::min(pos, length - 1));
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
    VectorDims dim5(5, 1LU);
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
        dim5[3] = 1LU;
    }
    return dim5;
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0F, 1 - std::abs(x));
}

class InterpolateExecutor {
public:
    static constexpr size_t DATA_ID = 0;
    static constexpr size_t TARGET_SHAPE_ID = 1;
    static constexpr size_t SCALES_ID = 2;
    static constexpr size_t AXES_ID = 3;
    static constexpr int CUBIC_GRID_LEN = 4;
    InterpolateExecutor(ExecutorContext::CPtr context) : _context(std::move(context)) {}

    virtual bool init(const InterpolateAttrs& interpolateAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr);
    virtual void exec(const std::vector<MemoryCPtr>& src,
                      const std::vector<MemoryPtr>& dst,
                      const void* post_ops_data_) = 0;
    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;

    virtual ~InterpolateExecutor() = default;
    [[nodiscard]] VectorDims getSrcDimPad5d() const {
        return srcDimPad5d;
    }
    const uint8_t* padPreprocess(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst);

private:
    void buildTblNN(const VectorDims& srcDimPad5d,
                    const VectorDims& dstDim5d,
                    const std::vector<float>& dataScales,
                    InterpolateLayoutType layout,
                    InterpolateNearestMode nearestMode);
    void buildTblLinearOnnx(const VectorDims& srcDimPad5d,
                            const VectorDims& dstDim5d,
                            const std::vector<float>& dataScales,
                            InterpolateLayoutType layout);
    void buildTblLinear(const VectorDims& srcDimPad5d,
                        const VectorDims& dstDim5d,
                        const std::vector<float>& dataScales,
                        int kernel_width,
                        bool antialias);
    void buildTblCubic(const VectorDims& srcDimPad5d,
                       const VectorDims& dstDim5d,
                       const std::vector<float>& dataScales,
                       float cubicCoeff,
                       InterpolateLayoutType layout);

    [[nodiscard]] float coordTransToInput(int outCoord, float scale, int inShape, int outShape) const;
    [[nodiscard]] static int nearestRound(float origin, bool isDownsample, InterpolateNearestMode nearestMode);
    void linearOnnxCF(int outCoord,
                      float scale,
                      int inShape,
                      int outShape,
                      int& index0,
                      int& index1,
                      float& weight0,
                      float& weight1);
    static std::vector<float> getCubicCoeffs(float mantissa, float a);

protected:
    InterpolateAttrs interpAttrs;
    VectorDims srcDimPad5d, dstDim5d;
    size_t srcDataSize = 0UL, dstDataSize = 0UL;
    int spatialDimSize = 0;
    size_t dataRank = 0UL;
    std::vector<int> indexTable;
    const ExecutorContext::CPtr _context;
};

using InterpolateExecutorPtr = std::shared_ptr<InterpolateExecutor>;
using InterpolateExecutorCPtr = std::shared_ptr<const InterpolateExecutor>;

class InterpolateExecutorBuilder {
public:
    virtual ~InterpolateExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const InterpolateAttrs& InterpolateAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual InterpolateExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using InterpolateExecutorBuilderPtr = std::shared_ptr<InterpolateExecutorBuilder>;
using InterpolateExecutorBuilderCPtr = std::shared_ptr<const InterpolateExecutorBuilder>;
}  // namespace ov::intel_cpu
