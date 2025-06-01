// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>
#include "nodes/executors/executor_config.hpp"

#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

enum : uint8_t { MAX_INPUT_INTERPOLATE = 8 };

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
    InterpolateLayoutType layout;
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

    static constexpr size_t DATA_ID = 0;
    static constexpr size_t TARGET_SHAPE_ID = 1;
    static constexpr size_t SCALES_ID = 2;
    static constexpr size_t AXES_ID = 3;
    static constexpr size_t SIZE_OR_SCALE_ID_V11 = 1;
    static constexpr size_t AXES_ID_V11 = 2;
    static constexpr int CUBIC_GRID_LEN = 4;
    static constexpr float PILLOW_BILINEAR_WINDOW_SCALE = 1.0f;
    static constexpr float PILLOW_BICUBIC_WINDOW_SCALE = 2.0f;
};

using InterpolateConfig = executor::Config<InterpolateAttrs>;

static inline bool isFloatCompatible(ov::element::Type prc) {
    return one_of(prc, ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::f64);
}

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

struct InterpolateKey {
    InterpolateAttrs nodeAttrs;
    VectorDims srcDims;
    VectorDims dstDims;
    std::vector<float> dataScales;
    dnnl::primitive_attr attr;

    [[nodiscard]] size_t hash() const;
    bool operator==(const InterpolateKey& rhs) const;
};

namespace legacy {

class InterpolateExecutorBaseLegacy {
public:
    InterpolateExecutorBaseLegacy(const InterpolateAttrs &interpAttrs,
                                  const VectorDims &srcDims,
                                  const VectorDims &dstDims,
                                  const std::vector<float> &dataScales);

    virtual void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) = 0;

    virtual ~InterpolateExecutorBaseLegacy() = default;

    [[nodiscard]] VectorDims getSrcDimPad5d() const {
        return srcDimPad5d;
    }

private:
    void buildTblNN(const VectorDims &srcDimPad5D,
                    const VectorDims &dstDim5D,
                    const std::vector<float> &dataScales,
                    InterpolateLayoutType layout,
                    InterpolateNearestMode nearestMode);

    void buildTblLinearOnnx(const VectorDims &srcDimPad5D,
                            const VectorDims &dstDim5D,
                            const std::vector<float> &dataScales,
                            InterpolateLayoutType layout);

    void buildTblLinear(const VectorDims &srcDimPad5D,
                        const VectorDims &dstDim5D,
                        const std::vector<float> &dataScales,
                        int kernel_width,
                        bool antialias);

    void buildTblCubic(const VectorDims &srcDimPad5D,
                       const VectorDims &dstDim5D,
                       const std::vector<float> &dataScales,
                       float cubicCoeff,
                       InterpolateLayoutType layout);

    void buildTblPillow(const VectorDims &srcDimPad5D,
                        const VectorDims &dstDim5D,
                        const std::vector<float> &dataScales,
                        float cubicCoeff,
                        InterpolateLayoutType layout);

    [[nodiscard]] float coordTransToInput(int outCoord, float scale, int inShape, int outShape) const;

    [[nodiscard]] int nearestRound(float origin, bool isDownsample, InterpolateNearestMode nearestMode) const;

    void linearOnnxCF(int outCoord,
                      float scale,
                      int inShape,
                      int outShape,
                      int &index0,
                      int &index1,
                      float &weight0,
                      float &weight1);

    std::vector<float> getCubicCoeffs(float mantissa, float a);

    static float getPillowBilinearCoeffs(float m);

    static float getPillowBicubicCoeffs(float m);

    inline void create_pillow_working_buf(InterpolateLayoutType layout);

protected:
    InterpolateAttrs baseInterpolateAttrs;
    InterpolateMode mode;
    InterpolateCoordTransMode coordTransMode;
    InterpolateLayoutType configured_for_layout;
    VectorDims srcDimPad5d, dstDim5d;
    ov::element::Type inputPrec, outputPrec;
    size_t srcDataSize, dstDataSize;
    size_t dataRank;
    int spatialDimSize;
    std::vector<int> auxTable;
    std::vector<uint8_t> pillow_working_buf;
    size_t m_threads_num = 0lu;
};

}  // namespace legacy

}  // namespace ov::intel_cpu