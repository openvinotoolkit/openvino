// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.hpp"

#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"

using namespace ov::intel_cpu;

bool ov::intel_cpu::InterpolateExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                              const std::vector<MemoryDescPtr>& srcDescs,
                                              const std::vector<MemoryDescPtr>& dstDescs,
                                              const dnnl::primitive_attr& attr) {
    const auto& srcDims = srcDescs[0]->getShape().getStaticDims();
    const auto& dstDims = dstDescs[0]->getShape().getStaticDims();
    interpAttrs = interpolateAttrs;
    srcDimPad5d = to5Dim(getPaddedInputShape(srcDims, interpolateAttrs.padBegin, interpolateAttrs.padEnd));
    dstDim5d = to5Dim(dstDims);
    srcDataSize = interpolateAttrs.inPrc.size();
    dstDataSize = interpolateAttrs.outPrc.size();
    dataRank = srcDims.size();
    spatialDimSize = getSpatialDimsNum(dataRank);

    switch (interpAttrs.mode) {
    case InterpolateMode::nearest: {
        buildTblNN(srcDimPad5d,
                   dstDim5d,
                   interpAttrs.dataScales,
                   interpolateAttrs.layout,
                   interpolateAttrs.nearestMode);
        break;
    }
    case InterpolateMode::linear_onnx: {
        buildTblLinearOnnx(srcDimPad5d, dstDim5d, interpAttrs.dataScales, interpolateAttrs.layout);
        break;
    }
    case InterpolateMode::linear: {
        static constexpr int LINEAR_KERNEL = 2;
        buildTblLinear(srcDimPad5d, dstDim5d, interpAttrs.dataScales, LINEAR_KERNEL, interpolateAttrs.antialias);
        break;
    }
    case InterpolateMode::cubic: {
        buildTblCubic(srcDimPad5d,
                      dstDim5d,
                      interpAttrs.dataScales,
                      interpolateAttrs.cubeCoeff,
                      interpolateAttrs.layout);
        break;
    }
    default: {
        OPENVINO_THROW("Interpolate executor does not support interpolate mode: ", interpAttrs.mode);
        break;
    }
    }
    return true;
}
// =====================================================================================================================
// index layout:
// d_0............d_OD-1, h_0..............h_OH-1, w_0................w_OW-1
void ov::intel_cpu::InterpolateExecutor::buildTblNN(const VectorDims& srcDimPad5d,
                                                    const VectorDims& dstDim5d,
                                                    const std::vector<float>& dataScales,
                                                    InterpolateLayoutType layout,
                                                    InterpolateNearestMode nearestMode) {
    const int dimSize = dataRank;
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.f;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    indexTable.resize(OD + OH + OW);
    bool isDDownsample = (fz < 1) ? true : false;
    bool isHDownsample = (fy < 1) ? true : false;
    bool isWDownsample = (fx < 1) ? true : false;
    for (int oz = 0; oz < static_cast<int>(OD); oz++) {
        float iz = coordTransToInput(oz, fz, ID, OD);
        indexTable[oz] = nearestRound(iz, isDDownsample, nearestMode);
        indexTable[oz] = clipCoord(indexTable[oz], ID);
    }
    for (int oy = 0; oy < static_cast<int>(OH); oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        indexTable[OD + oy] = nearestRound(iy, isHDownsample, nearestMode);
        indexTable[OD + oy] = clipCoord(indexTable[OD + oy], IH);
    }
    for (int ox = 0; ox < static_cast<int>(OW); ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        indexTable[OD + OH + ox] = nearestRound(ix, isWDownsample, nearestMode);
        indexTable[OD + OH + ox] = clipCoord(indexTable[OD + OH + ox], IW);
    }
}

// scale is float(outShape) / float(inShape)
// strictly consistent with onnx calc manner(div scale, not multiply inverse), given this is done offline
// the slight precison diff can produce obvious wrong value due to "nearest round" behavior for NN mode
float ov::intel_cpu::InterpolateExecutor::coordTransToInput(int outCoord,
                                                            float scale,
                                                            int inShape,
                                                            int outShape) const {
    if (scale == 1.0f || (inShape == outShape)) {
        return outCoord;
    }
    switch (interpAttrs.coordTransMode) {
    case InterpolateCoordTransMode::half_pixel: {
        return (outCoord + 0.5f) / scale - 0.5f;
    }
    case InterpolateCoordTransMode::pytorch_half_pixel: {
        if (outShape > 1) {
            return (outCoord + 0.5f) / scale - 0.5f;
        }
        return 0;
    }
    case InterpolateCoordTransMode::asymmetric: {
        return static_cast<float>(outCoord) / scale;
    }
    case InterpolateCoordTransMode::tf_half_pixel_for_nn: {
        return (outCoord + 0.5f) / scale;
    }
    case InterpolateCoordTransMode::align_corners: {
        if (outShape > 1) {
            return outCoord * (static_cast<float>(inShape - 1) / static_cast<float>(outShape - 1));
        }
        return 0;
    }
    default: {
        OPENVINO_THROW("Interpolate executor does not support specified coordinate transformation mode");
        break;
    }
    }
}

int ov::intel_cpu::InterpolateExecutor::nearestRound(float originCoord,
                                                     bool isDownsample,
                                                     InterpolateNearestMode nearestMode) const {
    switch (nearestMode) {
    case InterpolateNearestMode::round_prefer_floor: {
        if (originCoord == (static_cast<int>(originCoord) + 0.5f)) {
            return static_cast<int>(std::floor(originCoord));
        }
        return static_cast<int>(std::round(originCoord));
    }
    case InterpolateNearestMode::round_prefer_ceil: {
        return static_cast<int>(std::round(originCoord));
    }
    case InterpolateNearestMode::floor: {
        return static_cast<int>(std::floor(originCoord));
    }
    case InterpolateNearestMode::ceil: {
        return static_cast<int>(std::ceil(originCoord));
    }
    case InterpolateNearestMode::simple: {
        if (isDownsample) {
            return static_cast<int>(std::ceil(originCoord));
        }
        return static_cast<int>(originCoord);
    }
    default: {
        OPENVINO_THROW("Interpolate executor does not support specified nearest round mode");
        break;
    }
    }
}

void ov::intel_cpu::InterpolateExecutor::linearOnnxCF(int outCoord,
                                                      float scale,
                                                      int inShape,
                                                      int outShape,
                                                      int& index0,
                                                      int& index1,
                                                      float& weight0,
                                                      float& weight1) {
    float inCoord = coordTransToInput(outCoord, scale, inShape, outShape);
    inCoord = std::max(0.0f, std::min(inCoord, static_cast<float>(inShape - 1)));
    index0 = std::min(static_cast<int>(inCoord), inShape - 1);
    index1 = std::min(index0 + 1, inShape - 1);

    weight1 = std::fabs(inCoord - index0);
    weight0 = std::fabs(inCoord - index1);
    if (index0 == index1) {
        weight0 = 0.5f;
        weight1 = 0.5f;
    }
}

void ov::intel_cpu::InterpolateExecutor::buildTblLinearOnnx(const VectorDims& srcDimPad5d,
                                                            const VectorDims& dstDim5d,
                                                            const std::vector<float>& dataScales,
                                                            InterpolateLayoutType layout) {
    int dimSize = dataRank;
    float fz = (spatialDimSize > 2) ? dataScales[dimSize - 3] : 1.f;
    float fy = (spatialDimSize > 1) ? dataScales[dimSize - 2] : 1.f;
    float fx = dataScales[dimSize - 1];
    int ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, nullptr);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, nullptr);
    if (layout == InterpolateLayoutType::planar) {
        // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
        // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
        // weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5
        int eltInGrid = (spatialDimSize > 2) ? MAX_INPUT_INTERPOLATE : ((spatialDimSize > 1) ? 4 : 2);
        int idxType = 2;
        int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);
        indexTable.resize(idxType * scratchLen);

        indexPtr[0] = static_cast<int*>(&indexTable[0]);
        indexPtr[1] = static_cast<int*>(&indexTable[OW * OH * OD]);
        weightPtr[0] = reinterpret_cast<float*>(&indexTable[scratchLen]);
        weightPtr[1] = reinterpret_cast<float*>(&indexTable[scratchLen + OW * OH * OD]);
        if (spatialDimSize > 1) {
            indexPtr[2] = static_cast<int*>(&indexTable[2 * OW * OH * OD]);
            indexPtr[3] = static_cast<int*>(&indexTable[3 * OW * OH * OD]);
            weightPtr[2] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW * OH * OD]);
            weightPtr[3] = reinterpret_cast<float*>(&indexTable[scratchLen + 3 * OW * OH * OD]);
        }
        if (spatialDimSize > 2) {
            indexPtr[4] = static_cast<int*>(&indexTable[4 * OW * OH * OD]);
            indexPtr[5] = static_cast<int*>(&indexTable[5 * OW * OH * OD]);
            indexPtr[6] = static_cast<int*>(&indexTable[6 * OW * OH * OD]);
            indexPtr[7] = static_cast<int*>(&indexTable[7 * OW * OH * OD]);
            weightPtr[4] = reinterpret_cast<float*>(&indexTable[scratchLen + 4 * OW * OH * OD]);
            weightPtr[5] = reinterpret_cast<float*>(&indexTable[scratchLen + 5 * OW * OH * OD]);
        }
        int scale = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41) ? srcDataSize : 1;

        for (int oz = 0; oz < OD; oz++) {
            int izF, izE;
            float weightF, weightE;
            linearOnnxCF(oz, fz, ID, OD, izF, izE, weightF, weightE);
            int idxOz = oz * OH * OW;
            for (int oy = 0; oy < OH; oy++) {
                int iyT, iyB;
                float weightT, weightB;
                linearOnnxCF(oy, fy, IH, OH, iyT, iyB, weightT, weightB);
                int idxOzOy = idxOz + oy * OW;
                for (int ox = 0; ox < OW; ox++) {
                    int ixL, ixR;
                    float weightL, weightR;
                    linearOnnxCF(ox, fx, IW, OW, ixL, ixR, weightL, weightR);

                    int idxOzOyOx = idxOzOy + ox;
                    indexPtr[0][idxOzOyOx] = (izF * IH * IW + iyT * IW + ixL) * scale;
                    indexPtr[1][idxOzOyOx] = (izF * IH * IW + iyT * IW + ixR) * scale;
                    weightPtr[0][idxOzOyOx] = weightL;
                    weightPtr[1][idxOzOyOx] = weightR;
                    if (spatialDimSize > 1) {
                        indexPtr[2][idxOzOyOx] = (izF * IH * IW + iyB * IW + ixL) * scale;
                        indexPtr[3][idxOzOyOx] = (izF * IH * IW + iyB * IW + ixR) * scale;
                        weightPtr[2][idxOzOyOx] = weightT;
                        weightPtr[3][idxOzOyOx] = weightB;
                    }
                    if (spatialDimSize > 2) {
                        indexPtr[4][idxOzOyOx] = (izE * IH * IW + iyT * IW + ixL) * scale;
                        indexPtr[5][idxOzOyOx] = (izE * IH * IW + iyT * IW + ixR) * scale;
                        indexPtr[6][idxOzOyOx] = (izE * IH * IW + iyB * IW + ixL) * scale;
                        indexPtr[7][idxOzOyOx] = (izE * IH * IW + iyB * IW + ixR) * scale;
                        weightPtr[4][idxOzOyOx] = weightF;
                        weightPtr[5][idxOzOyOx] = weightE;
                    }
                }
            }
        }
    } else {
        // index: left:OW right:OW Top:OH Bottom:OH, Front:OD, End:OD
        // weight:same as index
        size_t scratchLen = rnd_up(OW + OW + OH + OH + OD + OD, 16);
        int idxType = 2;
        indexTable.resize(idxType * scratchLen);
        indexPtr[0] = static_cast<int*>(&indexTable[0]);
        indexPtr[1] = static_cast<int*>(&indexTable[OW]);
        indexPtr[2] = static_cast<int*>(&indexTable[2 * OW]);
        indexPtr[3] = static_cast<int*>(&indexTable[2 * OW + OH]);
        indexPtr[4] = static_cast<int*>(&indexTable[2 * OW + 2 * OH]);
        indexPtr[5] = static_cast<int*>(&indexTable[2 * OW + 2 * OH + OD]);

        weightPtr[0] = reinterpret_cast<float*>(&indexTable[scratchLen]);
        weightPtr[1] = reinterpret_cast<float*>(&indexTable[scratchLen + OW]);
        weightPtr[2] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW]);
        weightPtr[3] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW + OH]);
        weightPtr[4] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW + 2 * OH]);
        weightPtr[5] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW + 2 * OH + OD]);

        for (int ox = 0; ox < OW; ox++) {
            linearOnnxCF(ox, fx, IW, OW, indexPtr[0][ox], indexPtr[1][ox], weightPtr[0][ox], weightPtr[1][ox]);
        }
        for (int oy = 0; oy < OH; oy++) {
            linearOnnxCF(oy, fy, IH, OH, indexPtr[2][oy], indexPtr[3][oy], weightPtr[2][oy], weightPtr[3][oy]);
        }
        for (int oz = 0; oz < OD; oz++) {
            linearOnnxCF(oz, fz, ID, OD, indexPtr[4][oz], indexPtr[5][oz], weightPtr[4][oz], weightPtr[5][oz]);
        }
    }
}

// table layout:
// wd .........wd, wh............wh, ww.............ww, id...........id, ih............ih, iw..............iw
//                        |                                                      |
//                   wh0.....wh_diameter                                    ih0.....ih_diameter
void ov::intel_cpu::InterpolateExecutor::buildTblLinear(const VectorDims& srcDimPad5d,
                                                        const VectorDims& dstDim5d,
                                                        const std::vector<float>& dataScales,
                                                        int kernel_width,
                                                        bool antialias) {
    int dimSize = dataRank;
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.f;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    if (!(IW == OW && IH == OH && ID == OD)) {
        float ax = antialias ? fx : 1.0f;
        float ay = antialias ? fy : 1.0f;
        float az = antialias ? fz : 1.0f;

        int rx = (fx > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
        int ry = (fy > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
        int rz = (fz > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

        int diaOD = 2 * rz + 1;
        int diaOH = 2 * ry + 1;
        int diaOW = 2 * rx + 1;
        int sizeOD = OD * diaOD;
        int sizeOH = OH * diaOH;
        int sizeOW = OW * diaOW;
        indexTable.resize((sizeOD + sizeOH + sizeOW) * 2);
        auto* weightTable = reinterpret_cast<float*>(&indexTable[0]);
        auto* weightOD = static_cast<float*>(&weightTable[0]);
        auto* weightOH = static_cast<float*>(&weightTable[sizeOD]);
        auto* weightOW = static_cast<float*>(&weightTable[sizeOD + sizeOH]);

        auto* idxTable = static_cast<int*>(&indexTable[sizeOD + sizeOH + sizeOW]);
        auto* idxOD = static_cast<int*>(&idxTable[0]);
        auto* idxOH = static_cast<int*>(&idxTable[sizeOD]);
        auto* idxOW = static_cast<int*>(&idxTable[sizeOD + sizeOH]);

        for (int oz = 0; oz < static_cast<int>(OD); oz++) {
            float iz = coordTransToInput(oz, fz, ID, OD);
            auto iz_r = static_cast<int>(std::round(iz));
            for (int r = iz_r - rz, i = 0; r <= iz_r + rz; r++, i++) {
                idxOD[oz * diaOD + i] = r;
                if (r < 0 || r >= static_cast<int>(ID)) {
                    weightOD[oz * diaOD + i] = 0.f;
                } else {
                    float dz = iz - r;
                    weightOD[oz * diaOD + i] = az * triangleCoeff(az * dz);
                }
            }
        }
        for (int oy = 0; oy < static_cast<int>(OH); oy++) {
            float iy = coordTransToInput(oy, fy, IH, OH);
            auto iy_r = static_cast<int>(std::round(iy));
            for (int r = iy_r - ry, i = 0; r <= iy_r + ry; r++, i++) {
                idxOH[oy * diaOH + i] = r;
                if (r < 0 || r >= static_cast<int>(IH)) {
                    weightOH[oy * diaOH + i] = 0.f;
                } else {
                    float dy = iy - r;
                    weightOH[oy * diaOH + i] = ay * triangleCoeff(ay * dy);
                }
            }
        }
        for (int ox = 0; ox < static_cast<int>(OW); ox++) {
            float ix = coordTransToInput(ox, fx, IW, OW);
            auto ix_r = static_cast<int>(std::round(ix));
            for (int r = ix_r - rx, i = 0; r <= ix_r + rx; r++, i++) {
                idxOW[ox * diaOW + i] = r;
                if (r < 0 || r >= static_cast<int>(IW)) {
                    weightOW[ox * diaOW + i] = 0.f;
                } else {
                    float dx = ix - r;
                    weightOW[ox * diaOW + i] = ax * triangleCoeff(ax * dx);
                }
            }
        }
    }
}

std::vector<float> ov::intel_cpu::InterpolateExecutor::getCubicCoeffs(float mantissa, float a) {
    float m = std::fabs(mantissa);
    std::vector<float> coeffs(4, 0.f);

    coeffs[0] = a * (m - 1.0) * (m - 1.0) * m;
    coeffs[1] = ((a + 2.0) * m - (a + 3.0)) * m * m + 1.0;
    coeffs[2] = (((-a - 2.0) * m + (2.0 * a + 3.0)) * m - a) * m;
    coeffs[3] = -a * m * m * (m - 1.0);
    return coeffs;
}

// table layout:
// OW      OW         OW         OW         OW          OH       OH           OH           OH           OH
// x_idx   x_weight0  x_weight1  x_weight2  x_weight3   y_idx    y_weight0    y_weight1    y_weight2    y_weight3
void ov::intel_cpu::InterpolateExecutor::buildTblCubic(const VectorDims& srcDimPad5d,
                                                       const VectorDims& dstDim5d,
                                                       const std::vector<float>& dataScales,
                                                       float cubicCoeff,
                                                       InterpolateLayoutType layout) {
    int dimSize = dataRank;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    int IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OH = dstDim5d[3], OW = dstDim5d[4];

    // idxNum for index, CUBIC_GRID_LEN for weight
    const int idxNum = 1;
    size_t idxWeightSize = (CUBIC_GRID_LEN + idxNum) * OW + (CUBIC_GRID_LEN + idxNum) * OH;
    if (layout != InterpolateLayoutType::planar) {
        indexTable.resize(idxWeightSize);
    } else {
        size_t sequenceSize = 2 * OH * OW;
        indexTable.resize(idxWeightSize + sequenceSize);
    }

    int tblAdvance = 0;
    auto* xOrigin = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OW;
    auto* xFactor = reinterpret_cast<float*>(&indexTable[tblAdvance]);
    for (int ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        auto ix_r = static_cast<int>(std::floor(ix));
        xOrigin[ox] = ix_r;
        float m = ix - ix_r;
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        xFactor[CUBIC_GRID_LEN * ox] = coffes[0];
        xFactor[CUBIC_GRID_LEN * ox + 1] = coffes[1];
        xFactor[CUBIC_GRID_LEN * ox + 2] = coffes[2];
        xFactor[CUBIC_GRID_LEN * ox + 3] = coffes[3];
    }

    tblAdvance += CUBIC_GRID_LEN * OW;
    auto* yOrigin = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OH;
    auto* yFactor = reinterpret_cast<float*>(&indexTable[tblAdvance]);
    for (int oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        auto iy_r = static_cast<int>(std::floor(iy));
        yOrigin[oy] = iy_r;
        float m = iy - iy_r;
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        yFactor[CUBIC_GRID_LEN * oy] = coffes[0];
        yFactor[CUBIC_GRID_LEN * oy + 1] = coffes[1];
        yFactor[CUBIC_GRID_LEN * oy + 2] = coffes[2];
        yFactor[CUBIC_GRID_LEN * oy + 3] = coffes[3];
    }

    if (layout == InterpolateLayoutType::planar) {
        tblAdvance += CUBIC_GRID_LEN * OH;
        auto* sequenceOH = static_cast<int*>(&indexTable[tblAdvance]);
        tblAdvance += OH * OW;
        auto* sequenceOW = static_cast<int*>(&indexTable[tblAdvance]);
        for (int h = 0; h < OH; ++h) {
            int offset = h * OW;
            for (int w = 0; w < OW; ++w) {
                sequenceOH[offset + w] = h * sizeof(int);
                sequenceOW[offset + w] = w * sizeof(int);
            }
        }
    }
}

// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
inline VectorDims getBlockND(const VectorDims& shape) {
    int shapeRank = shape.size();
    VectorDims blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i + 1];
    }
    return blockND;
}

const uint8_t* ov::intel_cpu::InterpolateExecutor::padPreprocess(const std::vector<MemoryCPtr>& src,
                                                                 const std::vector<MemoryPtr>& dst) {
    const uint8_t* src_data_origin = src[0]->getDataAs<uint8_t>();

    const auto& srcDim = src[0]->getStaticDims();
    const auto& dstDim = dst[0]->getStaticDims();
    size_t dimSize = srcDim.size();
    auto srcDimPad = getSrcDimPad5d();

    const auto srcDim5d = to5Dim(srcDim);
    const auto srcDimPad5d = to5Dim(srcDimPad);
    const auto dstDim5d = to5Dim(dstDim);
    const auto srcDataSize = src[0]->getDesc().getPrecision().size();

    const uint8_t* src_data = nullptr;
    std::vector<uint8_t> srcPadded;
    if (interpAttrs.hasPad) {
        int padB0 = (dimSize > 2) ? interpAttrs.padBegin[0] : 0;
        int padB1 = (dimSize > 2) ? interpAttrs.padBegin[1] : 0;
        int padB2 = (dimSize == 5) ? interpAttrs.padBegin[dimSize - 3] : 0;
        int padB3 = interpAttrs.padBegin[dimSize - 2];
        int padB4 = interpAttrs.padBegin[dimSize - 1];

        VectorDims inShapeBlock = getBlockND(srcDim5d);
        VectorDims inShapePadBlock = getBlockND(srcDimPad5d);

        if (interpAttrs.layout == InterpolateLayoutType::planar) {
            srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
            auto* src_data_pad = static_cast<uint8_t*>(&srcPadded[0]);
            parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], [&](int n, int c, int d, int h) {
                const uint8_t* src = src_data_origin + (inShapeBlock[1] * n + inShapeBlock[2] * c +
                                                        inShapeBlock[3] * d + inShapeBlock[4] * h) *
                                                           srcDataSize;
                uint8_t* srcPad =
                    src_data_pad + (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                                    inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) *
                                       srcDataSize;
                cpu_memcpy(srcPad, src, srcDim5d[4] * srcDataSize);
            });
            src_data = src_data_pad;
        } else if (interpAttrs.layout == InterpolateLayoutType::by_channel) {
            srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
            auto* src_data_pad = static_cast<uint8_t*>(&srcPadded[0]);
            parallel_for4d(srcDim5d[0], srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int d, int h, int w) {
                const uint8_t* src = src_data_origin +
                                     (inShapeBlock[1] * n +
                                      (inShapeBlock[3] * d + inShapeBlock[4] * h + inShapeBlock[5] * w) * srcDim5d[1]) *
                                         srcDataSize;
                uint8_t* srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) +
                                                  (inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) +
                                                   inShapePadBlock[5] * (w + padB4)) *
                                                      srcDimPad5d[1] +
                                                  padB1) *
                                                     srcDataSize;
                cpu_memcpy(srcPad, src, srcDim5d[1] * srcDataSize);
            });
            src_data = src_data_pad;
        } else if (interpAttrs.layout == InterpolateLayoutType::block) {
            size_t blkSize = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 16 : 8;
            size_t CB = div_up(srcDimPad5d[1], blkSize);
            size_t eltsTotal = srcDimPad5d[0] * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize;
            srcPadded.resize(eltsTotal * srcDataSize, 0x0);
            auto* src_data_pad = static_cast<uint8_t*>(&srcPadded[0]);
            if ((srcDim5d[0] != srcDimPad5d[0]) || (srcDim5d[1] != srcDimPad5d[1])) {
                OPENVINO_THROW("Interpolate executor does not support padding on batch and channel dimensions");
            }
            parallel_for5d(
                srcDim5d[0],
                CB,
                srcDim5d[2],
                srcDim5d[3],
                srcDim5d[4],
                [&](int n, int cb, int d, int h, int w) {
                    const uint8_t* src = src_data_origin +
                                         (n * CB * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                         (cb * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                         (d * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                         (h * srcDim5d[4] * blkSize) * srcDataSize + (w * blkSize) * srcDataSize;
                    uint8_t* srcPad =
                        src_data_pad +
                        (n * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize +
                        (cb * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize +
                        ((d + padB2) * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize +
                        ((h + padB3) * srcDimPad5d[4] * blkSize) * srcDataSize + ((w + padB4) * blkSize) * srcDataSize;
                    cpu_memcpy(srcPad, src, blkSize * srcDataSize);
                });
            src_data = src_data_pad;
        }
    } else {
        src_data = src_data_origin;
    }
    return src_data;
}
