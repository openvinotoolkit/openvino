#include "ref_interpolate.hpp"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/primitive_attr.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "nodes/common/cpu_memcpy.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/precision_support.h"

namespace ov::intel_cpu {
void InterpolateRefExecutor::NNRef(const uint8_t* in_ptr_,
                                                uint8_t* out_ptr_,
                                                int B,
                                                int C,
                                                int ID,
                                                int IH,
                                                int IW,
                                                int OD,
                                                int OH,
                                                int OW) {
    auto* index_d = static_cast<int*>(auxTable.data());
    auto* index_h = static_cast<int*>(&auxTable[OD]);
    auto* index_w = static_cast<int*>(&auxTable[OD + OH]);

    const auto* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    auto* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);

    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const float* in_ptr = in_ptr_f32 + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]);
        float* out_ptr = out_ptr_f32 + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od);
        for (int oh = 0; oh < OH; oh++) {
            const float* in_ptr_h = in_ptr + (IW * index_h[oh]);
            float* out_ptr_h = out_ptr + (OW * oh);
            for (int ow = 0; ow < OW; ow++) {
                out_ptr_h[ow] = in_ptr_h[index_w[ow]];
            }
        }
    });
}

void InterpolateRefExecutor::linearOnnxRef(const uint8_t* in_ptr_,
                                                        uint8_t* out_ptr_,
                                                        int B,
                                                        int C,
                                                        int ID,
                                                        int IH,
                                                        int IW,
                                                        int OD,
                                                        int OH,
                                                        int OW) {
    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, nullptr);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, nullptr);
    // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
    // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
    // weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5

    int eltInGrid = [&]() {
        if (spatialDimSize > 2) {
            return MAX_INPUT_INTERPOLATE;
        }
        if (spatialDimSize > 1) {
            return 4;
        }
        return 2;
    }();
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);

    indexPtr[0] = static_cast<int*>(auxTable.data());
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

    const auto* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    auto* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);

    switch (spatialDimSize) {
    case 1:
        parallel_for4d(B, C, OD, OH, [&](size_t b, size_t c, size_t d, size_t h) {
            float* out_ptr_nc = out_ptr_f32 + (OD * OH * OW * C * b + OD * OH * OW * c + OH * OW * d + OW * h);
            const float* in_ptr_nc = in_ptr_f32 + (ID * IH * IW * C * b + ID * IH * IW * c + IH * IW * d + IW * h);
            for (int i = 0; i < OW; i++) {
                float src0 = in_ptr_nc[indexPtr[0][i]];
                float src1 = in_ptr_nc[indexPtr[1][i]];

                out_ptr_nc[i] = src0 * weightPtr[0][i] + src1 * weightPtr[1][i];
            }
        });
        break;
    case 2:
        parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t d) {
            float* out_ptr_nc = out_ptr_f32 + (OD * OH * OW * C * b + OD * OH * OW * c + OH * OW * d);
            const float* in_ptr_nc = in_ptr_f32 + (ID * IH * IW * C * b + ID * IH * IW * c + IH * IW * d);
            for (int i = 0; i < OH * OW; i++) {
                float src00 = in_ptr_nc[indexPtr[0][i]];
                float src01 = in_ptr_nc[indexPtr[1][i]];
                float src10 = in_ptr_nc[indexPtr[2][i]];
                float src11 = in_ptr_nc[indexPtr[3][i]];

                out_ptr_nc[i] = src00 * weightPtr[2][i] * weightPtr[0][i] + src01 * weightPtr[2][i] * weightPtr[1][i] +
                                src10 * weightPtr[3][i] * weightPtr[0][i] + src11 * weightPtr[3][i] * weightPtr[1][i];
            }
        });
        break;
    case 3:
        parallel_for2d(B, C, [&](size_t b, size_t c) {
            float* out_ptr_nc = out_ptr_f32 + (OD * OH * OW * C * b + OD * OH * OW * c);
            const float* in_ptr_nc = in_ptr_f32 + (ID * IH * IW * C * b + ID * IH * IW * c);
            for (int i = 0; i < OD * OH * OW; i++) {
                float src000 = in_ptr_nc[indexPtr[0][i]];
                float src001 = in_ptr_nc[indexPtr[1][i]];
                float src010 = in_ptr_nc[indexPtr[2][i]];
                float src011 = in_ptr_nc[indexPtr[3][i]];
                float src100 = in_ptr_nc[indexPtr[4][i]];
                float src101 = in_ptr_nc[indexPtr[5][i]];
                float src110 = in_ptr_nc[indexPtr[6][i]];
                float src111 = in_ptr_nc[indexPtr[7][i]];

                // float dstValue =
                // weightPtr[4][i] * weightPtr[2][i] * weightPtr[0][i] * src000 +
                // weightPtr[4][i] * weightPtr[2][i] * weightPtr[1][i] * src001 +
                // weightPtr[4][i] * weightPtr[3][i] * weightPtr[0][i] * src010 +
                // weightPtr[4][i] * weightPtr[3][i] * weightPtr[1][i] * src011 +
                // weightPtr[5][i] * weightPtr[2][i] * weightPtr[0][i] * src100 +
                // weightPtr[5][i] * weightPtr[2][i] * weightPtr[1][i] * src101 +
                // weightPtr[5][i] * weightPtr[3][i] * weightPtr[0][i] * src110 +
                // weightPtr[5][i] * weightPtr[3][i] * weightPtr[1][i] * src111;

                out_ptr_nc[i] =
                    weightPtr[4][i] * (weightPtr[2][i] * (weightPtr[0][i] * src000 + weightPtr[1][i] * src001) +
                                       weightPtr[3][i] * (weightPtr[0][i] * src010 + weightPtr[1][i] * src011)) +
                    weightPtr[5][i] * (weightPtr[2][i] * (weightPtr[0][i] * src100 + weightPtr[1][i] * src101) +
                                       weightPtr[3][i] * (weightPtr[0][i] * src110 + weightPtr[1][i] * src111));
            }
        });
        break;
    default:
        break;
    }
}

void InterpolateRefExecutor::cubicRef(const uint8_t* in_ptr_,
                                                   uint8_t* out_ptr_,
                                                   int B,
                                                   int C,
                                                   int IH,
                                                   int IW,
                                                   int OH,
                                                   int OW) {
    const int idxNum = 1;
    auto* xOrigin = static_cast<int*>(auxTable.data());
    auto* xFactor = reinterpret_cast<float*>(&auxTable[OW]);
    auto* yOrigin = static_cast<int*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    auto* yFactor = reinterpret_cast<float*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    const auto* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    auto* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);

    parallel_for4d(B, C, OH, OW, [&](size_t n, size_t c, size_t oy, size_t ox) {
        const float* in_ptr_nc = in_ptr_f32 + (IW * IH * C * n + IW * IH * c);
        float* out_ptr_nc = out_ptr_f32 + (OW * OH * C * n + OW * OH * c);

        int iy = yOrigin[oy];
        int ix = xOrigin[ox];

        float retY = 0.F;
        for (int y = iy - 1, i = 0; y <= iy + 2; y++, i++) {
            int yInRange = std::max(0, std::min(y, IH - 1));
            const float* in_ptr_nch = in_ptr_nc + IW * yInRange;
            float retX = 0.F;
            for (int x = ix - 1, j = 0; x <= ix + 2; x++, j++) {
                int xInRange = std::max(0, std::min(x, IW - 1));
                retX += xFactor[ox * CUBIC_GRID_LEN + j] * in_ptr_nch[xInRange];
            }
            retY += yFactor[oy * CUBIC_GRID_LEN + i] * retX;
        }
        out_ptr_nc[oy * OW + ox] = retY;
    });
}

float InterpolateRefExecutor::getValue(const uint8_t* base, size_t offset, ov::element::Type prec) {
    const uint8_t* baseOffset = base + offset;
    switch (prec) {
    case ov::element::u8: {
        return static_cast<float>(*baseOffset);
        break;
    }
    case ov::element::i8: {
        const auto* valuePtr = reinterpret_cast<const int8_t*>(baseOffset);
        return static_cast<float>(*valuePtr);
        break;
    }
    case ov::element::bf16: {
        const auto* valuePtr = reinterpret_cast<const uint16_t*>(baseOffset);
        return bfloat16_t::from_bits(*valuePtr);
        break;
    }
    case ov::element::f32: {
        const auto* valuePtr = reinterpret_cast<const float*>(baseOffset);
        return *valuePtr;
        break;
    }
    default: {
        OPENVINO_THROW("Interpolate layer does not support precision: ", prec);
        break;
    }
    }
}

void InterpolateRefExecutor::setValue(uint8_t* base, size_t offset, float value, ov::element::Type prec) {
    uint8_t* baseOffset = base + offset;
    switch (prec) {
    case ov::element::u8: {
        auto data = static_cast<uint8_t>(value < 0 ? 0 : value);
        cpu_memcpy(baseOffset, &data, 1);
        break;
    }
    case ov::element::i8: {
        auto data = static_cast<int8_t>(value);
        cpu_memcpy(baseOffset, &data, 1);
        break;
    }
    case ov::element::bf16: {
        uint16_t data = bfloat16_t(value).to_bits();
        cpu_memcpy(baseOffset, &data, 2);
        break;
    }
    case ov::element::f32: {
        cpu_memcpy(baseOffset, &value, sizeof(float));
        break;
    }
    default: {
        OPENVINO_THROW("Interpolate layer does not support precision: ", prec);
        break;
    }
    }
}

void InterpolateRefExecutor::linearInterpolation(const uint8_t* in_ptr_,
                                                              uint8_t* out_ptr_,
                                                              int B,
                                                              int C,
                                                              int ID,
                                                              int IH,
                                                              int IW,
                                                              float fx,
                                                              float fy,
                                                              float fz,
                                                              int OD,
                                                              int OH,
                                                              int OW,
                                                              int kernel_width,
                                                              bool antialias) {
    if (IW == OW && IH == OH && ID == OD) {
        size_t spatialDimSize = IW * IH * ID;
        // TODO: enable when fusing into interp with linear mode will support
        if (/*fusedWith.empty() &&*/ inputPrec == outputPrec) {
            size_t size = B * C * spatialDimSize * srcDataSize;
            cpu_memcpy(out_ptr_, in_ptr_, size);
        } else {
            parallel_for2d(B, C, [&](size_t b, size_t c) {
                const uint8_t* in_ptr_nc = in_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * srcDataSize;
                uint8_t* out_ptr_nc = out_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * dstDataSize;
                for (size_t i = 0; i < spatialDimSize; i++) {
                    float dstValue = getValue(in_ptr_nc, i * srcDataSize, inputPrec);
                    setValue(out_ptr_nc, i * dstDataSize, dstValue, outputPrec);
                }
            });
        }
        return;
    }

    float ax = antialias ? fx : 1.0F;
    float ay = antialias ? fy : 1.0F;
    float az = antialias ? fz : 1.0F;

    int rx = (fx > 1.0F) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
    int ry = (fy > 1.0F) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
    int rz = (fz > 1.0F) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

    int diaOD = 2 * rz + 1;
    int diaOH = 2 * ry + 1;
    int diaOW = 2 * rx + 1;
    int sizeOD = OD * diaOD;
    int sizeOH = OH * diaOH;
    int sizeOW = OW * diaOW;

    auto* weightTable = reinterpret_cast<float*>(auxTable.data());
    auto* weightOD = (&weightTable[0]);
    auto* weightOH = (&weightTable[sizeOD]);
    auto* weightOW = (&weightTable[sizeOD + sizeOH]);

    auto* idxTable = static_cast<int*>(&auxTable[sizeOD + sizeOH + sizeOW]);
    auto* idxOD = (&idxTable[0]);
    auto* idxOH = (&idxTable[sizeOD]);
    auto* idxOW = (&idxTable[sizeOD + sizeOH]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        const uint8_t* in_ptr_nc = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c) * srcDataSize;
        uint8_t* out_ptr_nc = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c) * dstDataSize;
        for (int oz = 0; oz < OD; oz++) {
            uint8_t* out_ptr_ncd = out_ptr_nc + (OW * OH * oz) * dstDataSize;
            for (int oy = 0; oy < OH; oy++) {
                uint8_t* out_ptr_ncdh = out_ptr_ncd + (OW * oy) * dstDataSize;
                for (int ox = 0; ox < OW; ox++) {
                    float sum = 0.F;
                    float wsum = 0.F;

                    // this comment explains the original algo.
                    // for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                    //    for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                    //        for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                    //            bool is_continue =  z < 0                     ||
                    //                                y < 0                     ||
                    //                                x < 0                     ||
                    //                                z >= static_cast<int>(ID) ||
                    //                                y >= static_cast<int>(IH) ||
                    //                                x >= static_cast<int>(IW);
                    //            if (is_continue)
                    //                continue;

                    //            float dx = ix - x;
                    //            float dy = iy - y;
                    //            float dz = iz - z;

                    //            float w = ax * triangleCoeff(ax * dx) *
                    //                      ay * triangleCoeff(ay * dy) *
                    //                      az * triangleCoeff(az * dz);

                    //            sum += w * getValue(in_ptr_nc, (z * IH * IW + y * IW + x) * srcDataSize, inputPrec);
                    //            wsum += w;
                    //        }
                    //    }
                    //}

                    for (int iz = 0; iz < diaOD; iz++) {
                        if (weightOD[oz * diaOD + iz] == 0.F) {
                            continue;
                        }
                        for (int iy = 0; iy < diaOH; iy++) {
                            if (weightOH[oy * diaOH + iy] == 0.F) {
                                continue;
                            }
                            for (int ix = 0; ix < diaOW; ix++) {
                                if (weightOW[ox * diaOW + ix] == 0.F) {
                                    continue;
                                }
                                float w =
                                    weightOD[oz * diaOD + iz] * weightOH[oy * diaOH + iy] * weightOW[ox * diaOW + ix];
                                float value = getValue(in_ptr_nc,
                                                       (idxOD[oz * diaOD + iz] * IH * IW + idxOH[oy * diaOH + iy] * IW +
                                                        idxOW[ox * diaOW + ix]) *
                                                           srcDataSize,
                                                       inputPrec);

                                sum += w * value;
                                wsum += w;
                            }
                        }
                    }

                    if (wsum == 0.0F) {
                        setValue(out_ptr_ncdh, ox * dstDataSize, 0.F, outputPrec);
                    } else {
                        float dst_value = sum / wsum;
                        setValue(out_ptr_ncdh, ox * dstDataSize, dst_value, outputPrec);
                    }
                }
            }
        }
    });
}

void InterpolateRefExecutor::pillowRef(const uint8_t* in_ptr_,
                                                    uint8_t* out_ptr_,
                                                    int B,
                                                    int C,
                                                    int IH,
                                                    int IW,
                                                    int OH,
                                                    int OW) {
    size_t offset = 0;
    int filterLenX = auxTable[offset];
    int filterLenY = auxTable[offset + 1];
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenX * OW;
    auto* weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenY * OH;
    auto* indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&auxTable[offset]);

    // workBuffer needed when both pass is true
    bool xPass = IW != OW;
    bool yPass = IH != OH;

    // --------    ----
    // |      |    |  |
    // |      |--> |  |
    // |      |    |  |
    // |      |    |  |
    // --------    ----
    //              \|/
    //             ----
    //             |  |
    //             |  |
    //             ----
    auto bc_loop = [&](size_t b, size_t c) {
        const uint8_t* in_ptr_nc = in_ptr_ + (IW * IH * C * b + IW * IH * c) * srcDataSize;
        uint8_t* out_ptr_nc = out_ptr_ + (OW * OH * C * b + OW * OH * c) * dstDataSize;
        uint8_t* xpass_out_ptr_nc = nullptr;
        const uint8_t* ypass_in_ptr_nc = nullptr;
        if (xPass && yPass) {
            size_t parallel_num = B * C;
            // IH * OW buf needed
            if (parallel_num < m_threads_num) {
                xpass_out_ptr_nc =
                    static_cast<uint8_t*>(&pillow_working_buf[(OW * IH * C * b + OW * IH * c) * srcDataSize]);
                ypass_in_ptr_nc =
                    static_cast<const uint8_t*>(&pillow_working_buf[(OW * IH * C * b + OW * IH * c) * srcDataSize]);
            } else {
                size_t threadsIdx = parallel_get_thread_num();
                auto buffer_size = static_cast<size_t>(OW) * IH;
                xpass_out_ptr_nc = static_cast<uint8_t*>(&pillow_working_buf[threadsIdx * buffer_size * srcDataSize]);
                ypass_in_ptr_nc =
                    static_cast<const uint8_t*>(&pillow_working_buf[threadsIdx * buffer_size * srcDataSize]);
            }
        } else if (xPass && !yPass) {
            xpass_out_ptr_nc = out_ptr_nc;
        } else if (!xPass && yPass) {
            ypass_in_ptr_nc = in_ptr_nc;
        } else if (!xPass && !yPass) {
            cpu_memcpy(out_ptr_nc, in_ptr_nc, OH * OW * dstDataSize);
        }
        float result = NAN;
        int f = 0;
        int filterS = 0;
        int filterL = 0;
        float* weight = nullptr;
        if (xPass) {
            for (size_t ih = 0; ih < static_cast<size_t>(IH); ih++) {
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    filterS = indexX[ow * 2];
                    filterL = std::min(indexX[ow * 2 + 1], filterLenX);
                    weight = (&weightX[ow * filterLenX]);
                    result = 0.F;
                    for (f = 0; f < filterL; f++) {
                        float pixel = getValue(in_ptr_nc, (ih * IW + f + filterS) * srcDataSize, inputPrec);
                        result += pixel * weight[f];
                    }
                    if (!isFloatCompatible(outputPrec)) {
                        result = static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5F : result - 0.5F));
                    }
                    // If Y pass follows, write to intermediate buffer (input precision/stride).
                    // Otherwise, write final result directly to output buffer (output precision/stride).
                    if (yPass) {
                        setValue(xpass_out_ptr_nc, (ih * OW + ow) * srcDataSize, result, inputPrec);
                    } else {
                        setValue(xpass_out_ptr_nc, (ih * OW + ow) * dstDataSize, result, outputPrec);
                    }
                }
            }
        }
        if (yPass) {
            for (size_t oh = 0; oh < static_cast<size_t>(OH); oh++) {
                filterS = indexY[oh * 2];
                filterL = std::min(indexY[oh * 2 + 1], filterLenY);
                weight = (&weightY[oh * filterLenY]);
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    result = 0.F;
                    for (f = 0; f < filterL; f++) {
                        float pixel = getValue(ypass_in_ptr_nc, ((f + filterS) * OW + ow) * srcDataSize, inputPrec);
                        result += pixel * weight[f];
                    }
                    if (!isFloatCompatible(outputPrec)) {
                        result = static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5F : result - 0.5F));
                    }
                    setValue(out_ptr_nc, (oh * OW + ow) * dstDataSize, result, outputPrec);
                }
            }
        }
    };

    parallel_nt_static(m_threads_num, [&](const int ithr, const int nthr) {
        for_2d(ithr, nthr, B, C, bc_loop);
    });
}

void InterpolateRefExecutor::pillowRefNCHWAsNHWC(const uint8_t* in_ptr_,
                                                              uint8_t* out_ptr_,
                                                              int B,
                                                              int C,
                                                              int IH,
                                                              int IW,
                                                              int OH,
                                                              int OW) {
    size_t offset = 0;
    int filterLenX = auxTable[offset];
    int filterLenY = auxTable[offset + 1];
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenX * OW;
    auto* weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenY * OH;
    auto* indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&auxTable[offset]);

    bool xPass = IW != OW;
    bool yPass = IH != OH;

    auto b_loop = [&](size_t b) {
        const uint8_t* in_ptr_b = in_ptr_ + b * IH * IW * C * srcDataSize;
        uint8_t* out_ptr_b = out_ptr_ + b * OH * OW * C * dstDataSize;

        uint8_t* xpass_out_ptr_b = nullptr;
        const uint8_t* ypass_in_ptr_b = nullptr;

        if (xPass && yPass) {
            size_t parallel_num = B;
            size_t buffer_size = static_cast<size_t>(IH) * OW * C;
            if (parallel_num < m_threads_num) {
                xpass_out_ptr_b = static_cast<uint8_t*>(&pillow_working_buf[b * buffer_size * srcDataSize]);
            } else {
                size_t threadsIdx = parallel_get_thread_num();
                xpass_out_ptr_b = static_cast<uint8_t*>(&pillow_working_buf[threadsIdx * buffer_size * srcDataSize]);
            }
            ypass_in_ptr_b = static_cast<const uint8_t*>(xpass_out_ptr_b);
        } else if (xPass && !yPass) {
            xpass_out_ptr_b = out_ptr_b;
        } else if (!xPass && yPass) {
            ypass_in_ptr_b = in_ptr_b;
        } else if (!xPass && !yPass) {
            cpu_memcpy(out_ptr_b, in_ptr_b, OH * OW * C * dstDataSize);
        }

        float result = NAN;
        int f = 0;
        int filterS = 0;
        int filterL = 0;
        float* weight = nullptr;

        if (xPass) {
            for (size_t ih = 0; ih < static_cast<size_t>(IH); ih++) {
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    filterS = indexX[ow * 2];
                    filterL = std::min(indexX[ow * 2 + 1], filterLenX);
                    weight = (&weightX[ow * filterLenX]);
                    for (size_t c = 0; c < static_cast<size_t>(C); c++) {
                        result = 0.F;
                        for (f = 0; f < filterL; f++) {
                            float pixel =
                                getValue(in_ptr_b, ((ih * IW + (f + filterS)) * C + c) * srcDataSize, inputPrec);
                            result += pixel * weight[f];
                        }
                        if (!isFloatCompatible(outputPrec)) {
                            result =
                                static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5F : result - 0.5F));
                        }
                        // If Y pass follows, write to intermediate buffer (input precision/stride).
                        // Otherwise, write final result directly to output buffer (output precision/stride).
                        if (yPass) {
                            setValue(xpass_out_ptr_b, ((ih * OW + ow) * C + c) * srcDataSize, result, inputPrec);
                        } else {
                            setValue(xpass_out_ptr_b, ((ih * OW + ow) * C + c) * dstDataSize, result, outputPrec);
                        }
                    }
                }
            }
        }

        if (yPass) {
            for (size_t oh = 0; oh < static_cast<size_t>(OH); oh++) {
                filterS = indexY[oh * 2];
                filterL = std::min(indexY[oh * 2 + 1], filterLenY);
                weight = (&weightY[oh * filterLenY]);
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    for (size_t c = 0; c < static_cast<size_t>(C); c++) {
                        result = 0.F;
                        for (f = 0; f < filterL; f++) {
                            float pixel =
                                getValue(ypass_in_ptr_b, (((f + filterS) * OW + ow) * C + c) * srcDataSize, inputPrec);
                            result += pixel * weight[f];
                        }
                        if (!isFloatCompatible(outputPrec)) {
                            result =
                                static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5F : result - 0.5F));
                        }
                        setValue(out_ptr_b, ((oh * OW + ow) * C + c) * dstDataSize, result, outputPrec);
                    }
                }
            }
        }
    };

    parallel_nt_static(m_threads_num, [&](const int ithr, const int nthr) {
        for_1d(ithr, nthr, B, b_loop);
    });
}

void InterpolateRefExecutor::exec(const uint8_t* in_ptr_,
                                               uint8_t* out_ptr_,
                                               [[maybe_unused]] const void* post_ops_data_) {
    size_t N = srcDimPad5d[0];
    size_t C = srcDimPad5d[1];
    size_t ID = srcDimPad5d[2];
    size_t IH = srcDimPad5d[3];
    size_t IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2];
    size_t OH = dstDim5d[3];
    size_t OW = dstDim5d[4];

    switch (mode) {
    case InterpolateMode::nearest: {
        NNRef(in_ptr_, out_ptr_, N, C, ID, IH, IW, OD, OH, OW);
        break;
    }
    case InterpolateMode::linear_onnx: {
        linearOnnxRef(in_ptr_, out_ptr_, N, C, ID, IH, IW, OD, OH, OW);
        break;
    }
    case InterpolateMode::cubic: {
        cubicRef(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
        break;
    }
    case InterpolateMode::linear: {
        float fz = (dataRank == 5) ? dataScales[dataRank - 3] : 1.F;
        float fy = dataScales[dataRank - 2];
        float fx = dataScales[dataRank - 1];

        bool isDownsample = (fx < 1.F) || (fy < 1.F) || (fz < 1.F);
        int kernel_width = 2;
        linearInterpolation(in_ptr_,
                            out_ptr_,
                            N,
                            C,
                            ID,
                            IH,
                            IW,
                            fx,
                            fy,
                            fz,
                            OD,
                            OH,
                            OW,
                            kernel_width,
                            isDownsample && antialias);
        break;
    }
    case InterpolateMode::bilinear_pillow:
    case InterpolateMode::bicubic_pillow: {
        if (refInterpAttrs.NCHWAsNHWC) {
            pillowRefNCHWAsNHWC(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
        } else {
            pillowRef(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
        }
        break;
    }
    default: {
        OPENVINO_THROW("Interpolate layer has unsupported interpolate mode: ", mode);
    }
    }
}
}
