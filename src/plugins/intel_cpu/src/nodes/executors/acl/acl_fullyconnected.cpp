// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_fullyconnected.hpp"

#include <cstdint>
#include <memory>

#include "cpu_memory.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "mlas/sgemm.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

#include "openvino/core/parallel.hpp"

#include "arm_neon.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"


namespace ov {
namespace intel_cpu {

using namespace executor;
using namespace dnnl;
using namespace ov::element;

bool ACLFCExecutor::supports(const FCConfig& config) {
    return true;
}

ACLFCExecutor::ACLFCExecutor(const FCAttrs& attrs,
                             const PostOps& postOps,
                             const MemoryArgs& memory,
                             const ExecutorContext::CPtr context)
    : m_attrs(attrs),
      m_memoryArgs(memory) {}

bool ACLFCExecutor::update(const MemoryArgs& memory) {
    return true;
}

template <typename T>
static std::vector<T> normalizeDimsTo2D(const std::vector<T>& dims) {
    return {std::accumulate(dims.begin(), dims.end() - 1, (T)1, std::multiplies<T>()), dims[dims.size() - 1]};
}

template <typename T>
static std::vector<T> squeezeDims(const std::vector<T>& dims) {
    T firstDim = dims[0];
    auto dimIterate = dims.begin();
    std::vector<T> restOfDims;
    while (firstDim == 1 && dimIterate != dims.end()) {
        firstDim *= (*(dimIterate++));
    }
    restOfDims.push_back(firstDim);
    while (dimIterate != dims.end()) {
        restOfDims.push_back(*(dimIterate++));
    }
    return restOfDims;
}

static int8_t get_u4(const uint8_t& val, bool high) {
    return high ? (val >> 4) : (val & 0xF);
}

typedef std::shared_ptr<const IMemory> shared_mem;

template<typename T1, typename T2, typename T3, typename T4>
void mm_exec(shared_mem src, shared_mem wei, shared_mem bias, shared_mem scales, shared_mem subtracts, shared_mem dst, bool weightsNonTransposed) {
    auto psrc = src->getDataAs<T1>();
    auto pwei = wei->getDataAs<T2>();
    auto pdst = dst->getDataAs<T1>();
    auto pscales = scales->getDataAs<T4>();
    T3* psub;
    std::vector<unsigned long> squeezedSubtracts;
    if (subtracts != nullptr) {
        psub = subtracts->getDataAs<T3>();
        squeezedSubtracts = squeezeDims(subtracts->getShape().getDims());
    }
    auto srcDims = normalizeDimsTo2D(src->getDesc().getShape().getDims());
    auto weiDims = wei->getDesc().getShape().getDims();
    auto scalesShape = scales->getDesc().getShape().getDims();
    auto M = srcDims[0];
    auto K = srcDims[1];
    auto N = weiDims[0];
    size_t scaleDimsBase = (scalesShape[0] != 1 || weightsNonTransposed) || scalesShape[1] == 1  ? 0 : 1;
    auto kGroups = weightsNonTransposed ? scalesShape[scaleDimsBase + 0] : scalesShape[scaleDimsBase + 1];
    auto kGroupSize = K / kGroups;

    std::cerr << src->getPrecision() << " " << wei->getPrecision() << " "
              << dst->getPrecision() << std::endl;
    std::cerr << M << " " << K << " " << N << std::endl;
    std::cerr << wei->getDesc().getShape().toString() << std::endl;
    std::cerr << src->getDesc().getShape().toString() << std::endl;
    std::cerr << scales->getShape().toString() << std::endl;
    std::cerr << kGroups << std::endl;
    std::cerr << kGroupSize << std::endl;
    std::cerr << subtracts->getShape().toString() << std::endl;
    std::cerr << subtracts->getPrecision() << std::endl;
    std::cerr << weightsNonTransposed << std::endl;

    const int BLOCK_SIZE = 32;
    int num_blocks_K = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::memset(pdst, 0, M * N * sizeof(T1));

//    parallel_for2d(M, N, [&](size_t m, size_t n) {
    parallel_for(num_blocks_N, [&](size_t n_block) {
        for (size_t k_block = 0; k_block < num_blocks_K; ++k_block) {
            size_t start_k = k_block * BLOCK_SIZE;
            size_t start_n = n_block * BLOCK_SIZE;
            size_t end_k = std::min(start_k + BLOCK_SIZE, K);
            size_t end_n = std::min(start_n + BLOCK_SIZE, N);

            size_t matrix_block_size_k = end_k - start_k;
            size_t matrix_block_size_n = end_n - start_n;
            arm_compute::Tensor a, b, c;
//            std::cout << matrix_block_size_n << " " << matrix_block_size_k << std::endl;
//            auto aInfo = arm_compute::TensorInfo(arm_compute::TensorShape(matrix_block_size_k, M), 1, arm_compute::DataType::F32);
//            auto bInfo = arm_compute::TensorInfo(arm_compute::TensorShape(matrix_block_size_k, matrix_block_size_n), 1, arm_compute::DataType::F32);
//            std::cout << matrix_block_size_k << " - " << matrix_block_size_n << std::endl;
//            auto cInfo = arm_compute::TensorInfo(arm_compute::TensorShape(matrix_block_size_n, M), 1, arm_compute::DataType::F32);
//            a.allocator()->init(aInfo);
//            b.allocator()->init(bInfo);
//            c.allocator()->init(cInfo);
//            std::cout << cInfo.strides_in_bytes()[1] << std::endl;
            auto aInfo = arm_compute::TensorInfo();
            aInfo.init(arm_compute::TensorShape(matrix_block_size_k, M),
                       arm_compute::Format::F32,
                       arm_compute::Strides{sizeof(float), K * sizeof(float)},
                       size_t(0),
                       size_t(M * K * sizeof(float)));
            auto bInfo = arm_compute::TensorInfo(arm_compute::TensorShape(matrix_block_size_k, matrix_block_size_n), 1, arm_compute::DataType::F32);
            auto cInfo = arm_compute::TensorInfo();
            cInfo.init(arm_compute::TensorShape(matrix_block_size_n, M),
                       arm_compute::Format::F32,
                       arm_compute::Strides{sizeof(float), N * sizeof(float)},
                       size_t(0),
                       size_t(N * M * sizeof(float)));
            a.allocator()->init(aInfo);
            b.allocator()->init(bInfo);
            c.allocator()->init(cInfo);
            b.allocator()->allocate();

            auto _b = reinterpret_cast<float*>(b.buffer());
            // TODO CHECK FOR TRANSPOSE CASE
            for (size_t n = start_n; n < end_n; ++n) {
                size_t k = start_k;
                for (; k + 8 <= end_k; k += 8) {
                    size_t weiIdx = weightsNonTransposed ? k * N + n : n * K + k;
                    size_t bufIdx = (n - start_n) * matrix_block_size_k + (k - start_k);
                    // size_t bufIdx = (k - start_k) * matrix_block_size_n + (n - start_n);

                    uint16x8_t quantized;
                    if (wei->getPrecision() == ov::element::u4) {
                        auto packed = vld1_u8(reinterpret_cast<const uint8_t*>(&pwei[weiIdx / 2]));
                        quantized = vmovl_u8(vzip_u8(vand_u8(packed, vdup_n_u8(0xF)), vshr_n_u8(packed, 4)).val[0]);
                    } else {
                        quantized = vmovl_u8(vld1_u8(reinterpret_cast<const uint8_t*>(&pwei[weiIdx])));
                    }
                    size_t scalesIdx = weightsNonTransposed ? (k / kGroupSize) * N + n : n * kGroups + (k / kGroupSize);

                    int16x8_t subtracted;
                    if (subtracts != nullptr) {
                        size_t subIdx = squeezedSubtracts[0] != 1 ? scalesIdx : 0;
                        if (subtracts->getPrecision() == ov::element::u4) {
                            subtracted = vmovl_s8(vdup_n_s8(get_u4(psub[subIdx / 2], subIdx % 2)));
                        } else {
                            subtracted = vmovl_u8(vdup_n_u8(psub[subIdx]));
                        }
                    } else {
                        subtracted = vdupq_n_u16(0);
                    }

                    int16x8_t zeroed = vsubq_s16(vreinterpretq_s16_u16(quantized), subtracted);
                    float32x4_t scales = vdupq_n_f32(pscales[scalesIdx]);
                    float32x4_t dequantized_low = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(zeroed))), scales);
                    float32x4_t dequantized_high = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(zeroed))), scales);

                    vst1q_f32(&_b[bufIdx], dequantized_low);
                    vst1q_f32(&_b[bufIdx + 4], dequantized_high);
                }
                for (; k < end_k; k++) {
                    size_t scalesIdx = weightsNonTransposed ? (k / kGroupSize) * N + n : n * kGroups + (k / kGroupSize);
                    size_t weiIdx = weightsNonTransposed ? k * N + n : n * K + k;
                    size_t bufIdx = (n - start_n) * matrix_block_size_k + (k - start_k);
                    size_t subIdx = 0;
                    if (subtracts)
                        subIdx =  squeezedSubtracts[0] != 1 ? scalesIdx : 0;
                    auto fscale = pscales[scalesIdx];
                    T2 fwei;
                    T3 fsub;
                    if (wei->getPrecision() == ov::element::u4) {
                        fwei = get_u4(pwei[weiIdx / 2], weiIdx % 2);
                    } else {
                        fwei = pwei[weiIdx];
                    }
                    if (subtracts != nullptr && subtracts->getPrecision() == ov::element::u4) {
                        fsub = get_u4(psub[subIdx / 2], subIdx % 2 == 1);
                    } else if (subtracts != nullptr) {
                        fsub = psub[subIdx];
                    }
                    float zerodWei;
                    if (subtracts != nullptr) {
                        zerodWei = static_cast<float>(fwei - fsub);
                    } else {
                        zerodWei = static_cast<float>(fwei);
                    }
                    _b[bufIdx] = (zerodWei * static_cast<float>(fscale));
                    // std::cout << k << " " << bufIdx << " " << weiIdx << " " << unsigned(fwei) << " " << fscale << " " << _b[bufIdx] << std::endl;
                }
            }
            // std::cout << src->getPrecision() << std::endl;
            auto _a = reinterpret_cast<void*>(psrc + start_k);
            auto _c = reinterpret_cast<void*>(pdst + start_n);
            a.allocator()->import_memory(_a);
            c.allocator()->import_memory(_c);

            arm_compute::NEGEMM gemm;
            arm_compute::GEMMInfo gemmInfo;
            gemmInfo.set_pretranspose_B(true);
            gemmInfo.set_accumulate(true);

//            arm_compute::Status val = gemm.validate(&aInfo, &bInfo, &cInfo, &cInfo, 1.0f, 1.0f, gemmInfo);
            gemm.configure(&a, &b, nullptr, &c, 1.0f, 0.0f, gemmInfo);
            gemm.run();
        }
    });
    if (bias != nullptr) {
        auto pbias = bias->getDataAs<const T1>();
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                pdst[m * N + n] += pbias[n];
            }
        }
    }

//    parallel_for2d(M, N, [&](size_t m, size_t n) {
//        size_t dstIdx = m * N + n;
//        T1 accum = 0.f;
//
//        for (size_t kb = 0; kb < kGroups; kb++) {
//            T1 groupAccum = 0.f;
//            for (size_t ki = 0; ki < kGroupSize; ki++) {
//                auto k = kb * kGroupSize + ki;
//                size_t srcIdx = m * K + k;
//                size_t weiIdx = weightsNonTransposed ? k * N + n : n * K + k;
//                accum += psrc[srcIdx] * _b[weiIdx];
//            }
//            accum += groupAccum;
//        }
//        if (bias != nullptr) {
//            accum += bias->getDataAs<const T1>()[n];
//        }
//        pdst[dstIdx] = accum;
//    });


//    parallel_for2d(M, N, [&](size_t m, size_t n) {
//            size_t dstIdx = m * N + n;
//            T1 accum = 0.f;
//
//            for (size_t kb = 0; kb < kGroups; kb++) {
//                T1 groupAccum = 0.f;
//                for (size_t ki = 0; ki < kGroupSize; ki++) {
//                    auto k = kb * kGroupSize + ki;
//                    size_t srcIdx = m * K + k;
//                    size_t weiIdx = weightsNonTransposed ? k * N + n : n * K + k;
//                    accum += psrc[srcIdx] * _b[weiIdx];
//                }
//                accum += groupAccum;
//            }
//            if (bias != nullptr) {
//                accum += bias->getDataAs<const T1>()[n];
//            }
//            pdst[dstIdx] = accum;
//    });

//    parallel_for2d(M, N, [&](size_t m, size_t n) {
//            size_t dstIdx = m * N + n;
//            T1 accum = 0.f;
//
//            for (size_t kb = 0; kb < kGroups; kb++) {
//                size_t scalesIdx = weightsNonTransposed ? kb * N + n : n * kGroups + kb;
//                size_t subIdx = 0;
//                if (subtracts)
//                    subIdx =  squeezedSubtracts[0] != 1 ? scalesIdx : 0;
//                auto fscale = pscales[scalesIdx];
//                T1 groupAccum = 0.f;
//                for (size_t ki = 0; ki < kGroupSize; ki++) {
//                    auto k = kb * kGroupSize + ki;
//                    size_t srcIdx = m * K + k;
//                    size_t weiIdx = weightsNonTransposed ? k * N + n : n * K + k;
//
//                    T2 fwei;
//                    T3 fsub;
//                    if (wei->getPrecision() == ov::element::u4) {
//                        fwei = get_u4(pwei[weiIdx / 2], weiIdx % 2);
//                    } else {
//                        fwei = pwei[weiIdx];
//                    }
//                    if (subtracts != nullptr && subtracts->getPrecision() == ov::element::u4) {
//                        fsub = get_u4(psub[subIdx / 2], subIdx % 2);
//                    } else if (subtracts != nullptr) {
//                        fsub = psub[subIdx];
//                    }
//                    float zerodWei;
//                    if (subtracts != nullptr) {
//                        zerodWei = static_cast<float>(fwei - fsub);
//                    } else {
//                        zerodWei = static_cast<float>(fwei);
//                    }
//
//                    groupAccum += psrc[srcIdx] * (zerodWei * static_cast<float>(fscale));
//                }
//                accum += groupAccum;
//            }
//            if (bias != nullptr) {
//                accum += bias->getDataAs<const T1>()[n];
//            }
//            pdst[dstIdx] = accum;
//    });
}

void ACLFCExecutor::execute(const MemoryArgs& memory) {
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    auto dst = memory.at(ARG_DST);
    std::shared_ptr<IMemory> bias = nullptr;
    if (m_attrs.withBias) {
        bias = memory.at(ARG_BIAS);
    }
    if (src->getPrecision() == ov::element::f32 && wei->getPrecision() == ov::element::u8) {
        if (m_attrs.decompressionSubtractPtr && m_attrs.decompressionSubtractPtr->getPrecision() == ov::element::u8) {
//            if (m_attrs.decompressionMultiplyPtr->getPrecision() == ov::element::f16)
//                mm_exec<float, uint8_t, uint8_t, ov::float16>(src, wei, bias, m_attrs.decompressionMultiplyPtr,
//                                                        m_attrs.decompressionSubtractPtr, dst, m_attrs.weightsNonTransposed);
//            else
                mm_exec<float, uint8_t, uint8_t, float>(src, wei, bias, m_attrs.decompressionMultiplyPtr,
                                                        m_attrs.decompressionSubtractPtr, dst, m_attrs.weightsNonTransposed);
        } else {
            mm_exec<float, uint8_t, float, float>(src, wei, bias, m_attrs.decompressionMultiplyPtr,
                                                  m_attrs.decompressionSubtractPtr, dst, m_attrs.weightsNonTransposed);
        }
    } else if (src->getPrecision() == ov::element::f32 && wei->getPrecision() == ov::element::i8) {
        if (m_attrs.decompressionSubtractPtr && m_attrs.decompressionSubtractPtr->getPrecision() == ov::element::i8) {
            mm_exec<float, int8_t, int8_t, float>(src, wei, bias, m_attrs.decompressionMultiplyPtr,
                                                  m_attrs.decompressionSubtractPtr, dst, m_attrs.weightsNonTransposed);
        } else {
            mm_exec<float, uint8_t, float, float>(src, wei, bias, m_attrs.decompressionMultiplyPtr,
                                                  m_attrs.decompressionSubtractPtr, dst, m_attrs.weightsNonTransposed);
        }
    } else if (src->getPrecision() == ov::element::f32 && wei->getPrecision() == ov::element::u4) {
        if (m_attrs.decompressionSubtractPtr && m_attrs.decompressionSubtractPtr->getPrecision() == ov::element::u4) {
            mm_exec<float, uint8_t, uint8_t, float>(src, wei, bias, m_attrs.decompressionMultiplyPtr,
                                                    m_attrs.decompressionSubtractPtr, dst, m_attrs.weightsNonTransposed);
        } else {
            mm_exec<float, uint8_t, float, float>(src, wei, bias, m_attrs.decompressionMultiplyPtr,
                                                  m_attrs.decompressionSubtractPtr, dst, m_attrs.weightsNonTransposed);
        }
    } else {
        std::cerr << src->getPrecision() << " " << wei->getPrecision() << " "
                  << (wei->getPrecision() == ov::element::u4) << std::endl;
    }
//
//    auto psrc = src->getDataAs<const float>();
//    auto pwei = wei->getDataAs<const uint8_t>();
//    auto pdst = dst->getDataAs<float>();
//    auto pscales = m_attrs.decompressionMultiplyPtr->getDataAs<const float>();
//    auto psub = m_attrs.decompressionSubtractPtr->getDataAs<const uint8_t>();
//
//
//    auto srcDims = normalizeDimsTo2D(src->getDesc().getShape().getDims());
//    auto weiDims = wei->getDesc().getShape().getDims();
//    auto scalesShape = m_attrs.decompressionMultiplyPtr->getDesc().getShape().getDims();
//
//    auto M = srcDims[0];
//    auto K = srcDims[1];
//    auto N = weiDims[0];
//    size_t scaleDimsBase = scalesShape[0] != 1 ? 0 : 1;
//    auto kGroups = m_attrs.weightsNonTransposed ? scalesShape[scaleDimsBase + 0] : scalesShape[scaleDimsBase + 1];
//    auto kGroupSize = K / kGroups;
//
//
//     for (size_t m = 0; m < M; m++) {
//         for (size_t n = 0; n < N; n++) {
//             size_t dstIdx = m * N + n;
//             pdst[dstIdx] = 0.f;
//
//             for (size_t kb = 0; kb < kGroups; kb++) {
//                 size_t scalesIdx = m_attrs.weightsNonTransposed ? kb * N + n : n * kGroups + kb;
//                 auto fscale = static_cast<float>(pscales[scalesIdx]);
//                 auto fsubtract = psub[scalesIdx];
//                 for (size_t ki = 0; ki < kGroupSize; ki++) {
//                     auto k = kb * kGroupSize + ki;
//                     size_t srcIdx = m * K + k;
//                     size_t weiIdx = m_attrs.weightsNonTransposed ? k * N + n : n * K + k;
//
//                     auto fwei = pwei[weiIdx];
//                     pdst[dstIdx] += psrc[srcIdx] * (static_cast<float>(fwei - fsubtract) * fscale);
//                 }
//             }
//             if (m_attrs.withBias) {
//                 pdst[dstIdx] += bias[n];
//             }
//         }
//     }
}

void ACLFCExecutor::moveMemToNumaNode(int numaNodeID) {
    if (curNumaNode == numaNodeID)
        return;
    curNumaNode = numaNodeID;
    mbind_move(packedWeights, numaNodeID);
    if (m_attrs.withBias) {
        mbind_move(m_memoryArgs.at(ARG_BIAS), numaNodeID);
    }
}

}  // namespace intel_cpu
}  // namespace ov
