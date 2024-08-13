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
    auto psrc = src->getDataAs<const T1>();
    auto pwei = wei->getDataAs<const T2>();
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

//    std::cerr << src->getPrecision() << " " << wei->getPrecision() << " "
//              << dst->getPrecision() << std::endl;
//    std::cerr << M << " " << K << " " << N << std::endl;
//    std::cerr << wei->getDesc().getShape().toString() << std::endl;
//    std::cerr << scales->getShape().toString() << std::endl;
//    std::cerr << kGroups << std::endl;
//    std::cerr << kGroupSize << std::endl;
//    std::cerr << subtracts->getShape().toString() << std::endl;
//    std::cerr << subtracts->getPrecision() << std::endl;
//    std::cerr << weightsNonTransposed << std::endl;

    parallel_for2d(M, N, [&](size_t m, size_t n) {
            size_t dstIdx = m * N + n;
            T1 accum = 0.f;

            for (size_t kb = 0; kb < kGroups; kb++) {
                size_t scalesIdx = weightsNonTransposed ? kb * N + n : n * kGroups + kb;
                size_t subIdx = 0;
                if (subtracts)
                    subIdx =  squeezedSubtracts[0] != 1 ? scalesIdx : 0;
                auto fscale = pscales[scalesIdx];
                T1 groupAccum = 0.f;
                for (size_t ki = 0; ki < kGroupSize; ki++) {
                    auto k = kb * kGroupSize + ki;
                    size_t srcIdx = m * K + k;
                    size_t weiIdx = weightsNonTransposed ? k * N + n : n * K + k;

                    T2 fwei;
                    T3 fsub;
                    if (wei->getPrecision() == ov::element::u4) {
                        fwei = get_u4(pwei[weiIdx / 2], weiIdx % 2);
                    } else {
                        fwei = pwei[weiIdx];
                    }
                    if (subtracts != nullptr && subtracts->getPrecision() == ov::element::u4) {
                        fsub = get_u4(psub[subIdx / 2], subIdx % 2);
                    } else if (subtracts != nullptr) {
                        fsub = psub[subIdx];
                    }
                    float zerodWei;
                    if (subtracts != nullptr) {
                        zerodWei = static_cast<float>(fwei - fsub);
                    } else {
                        zerodWei = static_cast<float>(fwei);
                    }

                    groupAccum += psrc[srcIdx] * (zerodWei * static_cast<float>(fscale));
                }
                accum += groupAccum;
            }
            if (bias != nullptr) {
                accum += bias->getDataAs<const T1>()[n];
            }
            pdst[dstIdx] = accum;
    });
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
            if (m_attrs.decompressionMultiplyPtr->getPrecision() == ov::element::f16)
                mm_exec<float, uint8_t, uint8_t, ov::float16>(src, wei, bias, m_attrs.decompressionMultiplyPtr,
                                                        m_attrs.decompressionSubtractPtr, dst, m_attrs.weightsNonTransposed);
            else
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
