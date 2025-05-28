// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate_config.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>
#include "openvino/core/parallel.hpp"
#include "common/primitive_hashing_utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

size_t InterpolateKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, nodeAttrs.mode);
    seed = hash_combine(seed, nodeAttrs.coordTransMode);
    seed = hash_combine(seed, nodeAttrs.nearestMode);
    seed = hash_combine(seed, nodeAttrs.layout);

    seed = hash_combine(seed, nodeAttrs.antialias);
    seed = hash_combine(seed, nodeAttrs.cubeCoeff);

    seed = get_vector_hash(seed, nodeAttrs.padBegin);
    seed = get_vector_hash(seed, nodeAttrs.padEnd);

    seed = hash_combine(seed, nodeAttrs.inPrc.hash());
    seed = hash_combine(seed, nodeAttrs.outPrc.hash());

    seed = get_vector_hash(seed, srcDims);
    seed = get_vector_hash(seed, dstDims);
    seed = get_vector_hash(seed, dataScales);

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool InterpolateKey::operator==(const InterpolateKey& rhs) const {
    if (nodeAttrs.mode != rhs.nodeAttrs.mode) {
        return false;
    }
    if (nodeAttrs.coordTransMode != rhs.nodeAttrs.coordTransMode) {
        return false;
    }
    if (nodeAttrs.nearestMode != rhs.nodeAttrs.nearestMode) {
        return false;
    }
    if (nodeAttrs.layout != rhs.nodeAttrs.layout) {
        return false;
    }
    if (nodeAttrs.antialias != rhs.nodeAttrs.antialias) {
        return false;
    }
    if (nodeAttrs.cubeCoeff != rhs.nodeAttrs.cubeCoeff) {
        return false;
    }
    if (nodeAttrs.padBegin != rhs.nodeAttrs.padBegin) {
        return false;
    }
    if (nodeAttrs.padEnd != rhs.nodeAttrs.padEnd) {
        return false;
    }
    if (nodeAttrs.inPrc != rhs.nodeAttrs.inPrc) {
        return false;
    }
    if (nodeAttrs.outPrc != rhs.nodeAttrs.outPrc) {
        return false;
    }

    if (srcDims != rhs.srcDims) {
        return false;
    }
    if (dstDims != rhs.dstDims) {
        return false;
    }
    if (dataScales != rhs.dataScales) {
        return false;
    }
    if (!(*attr.get() == *rhs.attr.get())) {
        return false;
    }

    return true;
}

// =====================================================================================================================
// index layout:
// d_0............d_OD-1, h_0..............h_OH-1, w_0................w_OW-1
void InterpolateExecutorBase::buildTblNN(const VectorDims& srcDimPad5d,
                                         const VectorDims& dstDim5d,
                                         const std::vector<float>& dataScales,
                                         [[maybe_unused]] InterpolateLayoutType layout,
                                         InterpolateNearestMode nearestMode) {
    const int dimSize = dataRank;
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.f;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    auxTable.resize(OD + OH + OW);
    bool isDDownsample = (fz < 1) ? true : false;
    bool isHDownsample = (fy < 1) ? true : false;
    bool isWDownsample = (fx < 1) ? true : false;
    for (size_t oz = 0; oz < OD; oz++) {
        float iz = coordTransToInput(oz, fz, ID, OD);
        auxTable[oz] = nearestRound(iz, isDDownsample, nearestMode);
        auxTable[oz] = clipCoord(auxTable[oz], ID);
    }
    for (size_t oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        auxTable[OD + oy] = nearestRound(iy, isHDownsample, nearestMode);
        auxTable[OD + oy] = clipCoord(auxTable[OD + oy], IH);
    }
    for (size_t ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        auxTable[OD + OH + ox] = nearestRound(ix, isWDownsample, nearestMode);
        auxTable[OD + OH + ox] = clipCoord(auxTable[OD + OH + ox], IW);
    }
}

// scale is float(outShape) / float(inShape)
// strictly consistent with onnx calc manner(div scale, not multiply inverse), given this is done offline
// the slight precison diff can produce obvious wrong value due to "nearest round" behavior for NN mode
float InterpolateExecutorBase::coordTransToInput(int outCoord,
                                                 float scale,
                                                 int inShape,
                                                 int outShape) const {
    if (scale == 1.0f || (inShape == outShape)) {
        return outCoord;
    }
    switch (coordTransMode) {
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
            OPENVINO_THROW("does not support specified coordinate transformation mode");
            break;
        }
    }
}

int InterpolateExecutorBase::nearestRound(float originCoord,
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
            OPENVINO_THROW("does not support specified nearest round mode");
            break;
        }
    }
}

void InterpolateExecutorBase::linearOnnxCF(int outCoord,
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

void InterpolateExecutorBase::buildTblLinearOnnx(const VectorDims& srcDimPad5d,
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
        auxTable.resize(idxType * scratchLen);

        indexPtr[0] = static_cast<int*>(&auxTable[0]);
        indexPtr[1] = static_cast<int*>(&auxTable[OW * OH * OD]);
        weightPtr[0] = reinterpret_cast<float*>(&auxTable[scratchLen]);
        weightPtr[1] = reinterpret_cast<float*>(&auxTable[scratchLen + OW * OH * OD]);
        if (spatialDimSize > 1) {
            indexPtr[2] = static_cast<int*>(&auxTable[2 * OW * OH * OD]);
            indexPtr[3] = static_cast<int*>(&auxTable[3 * OW * OH * OD]);
            weightPtr[2] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW * OH * OD]);
            weightPtr[3] = reinterpret_cast<float*>(&auxTable[scratchLen + 3 * OW * OH * OD]);
        }
        if (spatialDimSize > 2) {
            indexPtr[4] = static_cast<int*>(&auxTable[4 * OW * OH * OD]);
            indexPtr[5] = static_cast<int*>(&auxTable[5 * OW * OH * OD]);
            indexPtr[6] = static_cast<int*>(&auxTable[6 * OW * OH * OD]);
            indexPtr[7] = static_cast<int*>(&auxTable[7 * OW * OH * OD]);
            weightPtr[4] = reinterpret_cast<float*>(&auxTable[scratchLen + 4 * OW * OH * OD]);
            weightPtr[5] = reinterpret_cast<float*>(&auxTable[scratchLen + 5 * OW * OH * OD]);
        }
        int scale = mayiuse(cpu::x64::sse41) ? srcDataSize : 1;

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
        auxTable.resize(idxType * scratchLen);
        indexPtr[0] = static_cast<int*>(&auxTable[0]);
        indexPtr[1] = static_cast<int*>(&auxTable[OW]);
        indexPtr[2] = static_cast<int*>(&auxTable[2 * OW]);
        indexPtr[3] = static_cast<int*>(&auxTable[2 * OW + OH]);
        indexPtr[4] = static_cast<int*>(&auxTable[2 * OW + 2 * OH]);
        indexPtr[5] = static_cast<int*>(&auxTable[2 * OW + 2 * OH + OD]);

        weightPtr[0] = reinterpret_cast<float*>(&auxTable[scratchLen]);
        weightPtr[1] = reinterpret_cast<float*>(&auxTable[scratchLen + OW]);
        weightPtr[2] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW]);
        weightPtr[3] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + OH]);
        weightPtr[4] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + 2 * OH]);
        weightPtr[5] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + 2 * OH + OD]);

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
void InterpolateExecutorBase::buildTblLinear(const VectorDims& srcDimPad5d,
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
        auxTable.resize((sizeOD + sizeOH + sizeOW) * 2);
        auto* weightTable = reinterpret_cast<float*>(&auxTable[0]);
        auto* weightOD = static_cast<float*>(&weightTable[0]);
        auto* weightOH = static_cast<float*>(&weightTable[sizeOD]);
        auto* weightOW = static_cast<float*>(&weightTable[sizeOD + sizeOH]);

        auto* idxTable = static_cast<int*>(&auxTable[sizeOD + sizeOH + sizeOW]);
        auto* idxOD = static_cast<int*>(&idxTable[0]);
        auto* idxOH = static_cast<int*>(&idxTable[sizeOD]);
        auto* idxOW = static_cast<int*>(&idxTable[sizeOD + sizeOH]);

        for (size_t oz = 0; oz < OD; oz++) {
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
        for (size_t oy = 0; oy < OH; oy++) {
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
        for (size_t ox = 0; ox < OW; ox++) {
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

std::vector<float> InterpolateExecutorBase::getCubicCoeffs(float mantissa, float a) {
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
void InterpolateExecutorBase::buildTblCubic(const VectorDims& srcDimPad5d,
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
    size_t idxWeightSize = (baseInterpolateAttrs.CUBIC_GRID_LEN + idxNum) * OW + (baseInterpolateAttrs.CUBIC_GRID_LEN + idxNum) * OH;
    if (layout != InterpolateLayoutType::planar) {
        auxTable.resize(idxWeightSize);
    } else {
        size_t sequenceSize = 2 * OH * OW;
        auxTable.resize(idxWeightSize + sequenceSize);
    }

    int tblAdvance = 0;
    auto* xOrigin = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OW;
    auto* xFactor = reinterpret_cast<float*>(&auxTable[tblAdvance]);
    for (int ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        auto ix_r = static_cast<int>(std::floor(ix));
        xOrigin[ox] = ix_r;
        float m = ix - ix_r;
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        xFactor[baseInterpolateAttrs.CUBIC_GRID_LEN * ox] = coffes[0];
        xFactor[baseInterpolateAttrs.CUBIC_GRID_LEN * ox + 1] = coffes[1];
        xFactor[baseInterpolateAttrs.CUBIC_GRID_LEN * ox + 2] = coffes[2];
        xFactor[baseInterpolateAttrs.CUBIC_GRID_LEN * ox + 3] = coffes[3];
    }

    tblAdvance += baseInterpolateAttrs.CUBIC_GRID_LEN * OW;
    auto* yOrigin = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OH;
    auto* yFactor = reinterpret_cast<float*>(&auxTable[tblAdvance]);
    for (int oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        auto iy_r = static_cast<int>(std::floor(iy));
        yOrigin[oy] = iy_r;
        float m = iy - iy_r;
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        yFactor[baseInterpolateAttrs.CUBIC_GRID_LEN * oy] = coffes[0];
        yFactor[baseInterpolateAttrs.CUBIC_GRID_LEN * oy + 1] = coffes[1];
        yFactor[baseInterpolateAttrs.CUBIC_GRID_LEN * oy + 2] = coffes[2];
        yFactor[baseInterpolateAttrs.CUBIC_GRID_LEN * oy + 3] = coffes[3];
    }

    if (layout == InterpolateLayoutType::planar) {
        tblAdvance += baseInterpolateAttrs.CUBIC_GRID_LEN * OH;
        auto* sequenceOH = static_cast<int*>(&auxTable[tblAdvance]);
        tblAdvance += OH * OW;
        auto* sequenceOW = static_cast<int*>(&auxTable[tblAdvance]);
        for (int h = 0; h < OH; ++h) {
            int offset = h * OW;
            for (int w = 0; w < OW; ++w) {
                sequenceOH[offset + w] = h * sizeof(int);
                sequenceOW[offset + w] = w * sizeof(int);
            }
        }
    }
}

float InterpolateExecutorBase::getPillowBilinearCoeffs(float m) {
    if (m < 0.0f) {
        m = -m;
    }
    if (m < 1.0) {
        return 1.0f - m;
    }
    return 0.0f;
}

float InterpolateExecutorBase::getPillowBicubicCoeffs(float m) {
    float a = -0.5f;
    if (m < 0.0f) {
        m = -m;
    }
    if (m < 1.0) {
        return ((a + 2.0) * m - (a + 3.0)) * m * m + 1.0;
    }
    if (m < 2.0f) {
        return (((m - 5) * m + 8) * m - 4) * a;
    }
    return 0.0f;
}

void InterpolateExecutorBase::buildTblPillow(const VectorDims& srcDimPad5d,
                                             const VectorDims& dstDim5d,
                                             const std::vector<float>& dataScales,
                                             [[maybe_unused]] float cubicCoeff,
                                             [[maybe_unused]] InterpolateLayoutType layout) {
    int dimSize = dataRank;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    int IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OH = dstDim5d[3], OW = dstDim5d[4];

    struct filterArgs {
        float (*weightGen)(float m);
        float ScaleClipReciprocal;
        float filterRadius;
        float filterLen;
    };

    // pillowScale: e.g. 2.0 means down sample 2 times
    auto generateArgs = [&](float pillowScale) -> filterArgs {
        filterArgs args;
        float scaleClip = pillowScale < 1.0f ? 1.0f : pillowScale;
        args.ScaleClipReciprocal = 1.0f / scaleClip;
        args.filterRadius = (mode == InterpolateMode::bilinear_pillow) ? baseInterpolateAttrs.PILLOW_BILINEAR_WINDOW_SCALE * scaleClip
                                                                       : baseInterpolateAttrs.PILLOW_BICUBIC_WINDOW_SCALE * scaleClip;
        args.filterLen = static_cast<int>(std::ceil(args.filterRadius) * 2 + 1);
        args.weightGen =
                (mode == InterpolateMode::bilinear_pillow) ? this->getPillowBilinearCoeffs : this->getPillowBicubicCoeffs;
        return args;
    };

    filterArgs filterArgsX = generateArgs(1.0f / fx);
    filterArgs filterArgsY = generateArgs(1.0f / fy);

    // index with Run Length Coding(start+len for each ow/oh)
    size_t weightLen = filterArgsX.filterLen * OW + filterArgsY.filterLen * OH;
    size_t boundLen = 2 * OW + 2 * OH;
    auxTable.resize(2 + weightLen + boundLen);
    size_t offset = 0;
    auxTable[offset] = filterArgsX.filterLen;
    auxTable[offset + 1] = filterArgsY.filterLen;
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterArgsX.filterLen * OW;
    auto* weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterArgsY.filterLen * OH;
    auto* indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&auxTable[offset]);

    auto generateTbl = [&](int inLen, int outLen, float fScale, filterArgs args, float* weightTbl, int* idxTbl) {
        int min = 0;
        int max = 0;
        for (int ox = 0; ox < outLen; ox++) {
            float ixCenter = coordTransToInput(ox, fScale, inLen, outLen);
            min = static_cast<int>(ixCenter - args.filterRadius + 0.5f);
            if (min < 0) {
                min = 0;
            }
            max = static_cast<int>(ixCenter + args.filterRadius + 0.5f);
            if (max > inLen) {
                max = inLen;
            }
            // use [min, max) range of input to get output
            // below let max become len
            max -= min;
            idxTbl[2 * ox] = min;
            idxTbl[2 * ox + 1] = max;

            size_t offset = ox * args.filterLen;
            float weightSum = 0;
            int ix = 0;
            for (ix = 0; ix < max; ix++) {
                // use distance to center as a parameter to compute weight
                float w = args.weightGen((ix + min - ixCenter + 0.5) * args.ScaleClipReciprocal);
                weightTbl[offset + ix] = w;
                weightSum += w;
            }
            if (weightSum != 0) {
                for (ix = 0; ix < max; ix++) {
                    weightTbl[offset + ix] /= weightSum;
                }
            }

            // filterlen is maximum possible len, set others to 0 for possible uniform process(vector)
            for (; ix < args.filterLen; ix++) {
                weightTbl[offset + ix] = 0.f;
            }
        }
    };

    generateTbl(IW, OW, fx, filterArgsX, weightX, indexX);
    generateTbl(IH, OH, fy, filterArgsY, weightY, indexY);
}


void InterpolateExecutorBase::create_pillow_working_buf(InterpolateLayoutType layout) {
    if (srcDimPad5d[3] == dstDim5d[3] || srcDimPad5d[4] == dstDim5d[4]) {
        return;
    }
    size_t bufSize = srcDimPad5d[3] * dstDim5d[4] * srcDataSize;  // IH * OW
    m_threads_num = parallel_get_max_threads();
    if (layout == InterpolateLayoutType::planar) {
        // B and C execute in parallel, need separate buf
        size_t parallel_num = srcDimPad5d[0] * srcDimPad5d[1];
        bufSize *= std::min(m_threads_num, parallel_num);
    } else {
        bufSize *= srcDimPad5d[1];  // *C
        // B execute in parallel, need separate buf
        size_t parallel_num = srcDimPad5d[0];
        bufSize *= std::min(m_threads_num, parallel_num);
    }
    pillow_working_buf.resize(bufSize);
}

InterpolateExecutorBase::InterpolateExecutorBase(const InterpolateAttrs& interpAttrs,
                                                 const VectorDims& srcDims,
                                                 const VectorDims& dstDims,
                                                 const std::vector<float>& dataScales)
        : baseInterpolateAttrs(interpAttrs), 
          mode(interpAttrs.mode),
          coordTransMode(interpAttrs.coordTransMode),
          configured_for_layout(interpAttrs.layout),
          srcDimPad5d(to5Dim(getPaddedInputShape(srcDims, interpAttrs.padBegin, interpAttrs.padEnd))),
          dstDim5d(to5Dim(dstDims)),
          inputPrec(interpAttrs.inPrc),
          outputPrec(interpAttrs.outPrc),
          srcDataSize(interpAttrs.inPrc.size()),
          dstDataSize(interpAttrs.outPrc.size()),
          dataRank(srcDims.size()),
          spatialDimSize(getSpatialDimsNum(dataRank)) {
    switch (mode) {
        case InterpolateMode::nearest: {
            buildTblNN(srcDimPad5d, dstDim5d, dataScales, interpAttrs.layout, interpAttrs.nearestMode);
            break;
        }
        case InterpolateMode::linear_onnx: {
            buildTblLinearOnnx(srcDimPad5d, dstDim5d, dataScales, interpAttrs.layout);
            break;
        }
        case InterpolateMode::linear: {
            static constexpr int LINEAR_KERNEL = 2;
            buildTblLinear(srcDimPad5d, dstDim5d, dataScales, LINEAR_KERNEL, interpAttrs.antialias);
            break;
        }
        case InterpolateMode::cubic: {
            buildTblCubic(srcDimPad5d, dstDim5d, dataScales, interpAttrs.cubeCoeff, interpAttrs.layout);
            break;
        }
        case InterpolateMode::bilinear_pillow:
        case InterpolateMode::bicubic_pillow: {
            buildTblPillow(srcDimPad5d, dstDim5d, dataScales, interpAttrs.cubeCoeff, interpAttrs.layout);
            if ((srcDimPad5d[4] != dstDim5d[4]) && (srcDimPad5d[3] != dstDim5d[3])) {
                create_pillow_working_buf(interpAttrs.layout);
            }
            break;
        }
        default: {
            OPENVINO_THROW("Interpolate executor does not support interpolate mode: ", mode);
            break;
        }
    }
}

}  // namespace ov::intel_cpu
