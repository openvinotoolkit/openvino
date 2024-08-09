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

static int8_t get_u4(const uint8_t& val, bool high) {
    return high ? (val >> 4) : (val & 0xF);
}

void ACLFCExecutor::execute(const MemoryArgs& memory) {
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    auto dst = memory.at(ARG_DST);

    auto psrc = src->getDataAs<const float>();
    auto pwei = wei->getDataAs<const uint8_t>();
    auto pdst = dst->getDataAs<float>();
    std::cout << wei->getDescPtr()->getPrecision() << std::endl;
    std::cout << m_attrs.decompressionMultiplyPtr << std::endl;
//    auto pscales = m_attrs.decompressionMultiplyPtr->getDataAs<const float8_e8m0>();
//
//    auto srcDims = normalizeDimsTo2D(src->getDesc().getShape().getDims());
//    auto weiDims = wei->getDesc().getShape().getDims();
//    auto scalesShape = m_attrs.decompressionMultiplyPtr->getDesc().getShape().getDims();
//
//    auto M = srcDims[0];
//    auto K = srcDims[1];
//    auto N = weiDims[0];
//    auto kGroups = m_attrs.weightsNonTransposed ? scalesShape[0] : scalesShape[1];
//    auto kGroupSize = K / kGroups;
//
//    std::cerr << M << " " << K << " " << N << std::endl;
//    std::cerr << scalesShape[0] << " " << scalesShape[1] << " " << scalesShape[2] << std::endl;
//
//    // for (size_t m = 0; m < M; m++) {
//    //     for (size_t n = 0; n < N; n++) {
//    parallel_for2d(M, N, [&](size_t m, size_t n) {
//        size_t dstIdx = m * N + n;
//        pdst[dstIdx] = 0.f;
//
//        for (size_t kb = 0; kb < kGroups; kb++) {
//            size_t scalesIdx = m_attrs.weightsNonTransposed ? kb * N + n : n * kGroups + kb;
//            auto fscale = static_cast<float>(pscales[scalesIdx]);
//
//            for (size_t ki = 0; ki < kGroupSize; ki++) {
//                auto k = kb * kGroupSize + ki;
//                size_t srcIdx = m * K + k;
//                size_t weiIdx = m_attrs.weightsNonTransposed ? k * N + n : n * K + k;
//
//                auto fwei = static_cast<float>(float4_e2m1::from_bits(get_u4(pwei[weiIdx / 2], weiIdx % 2)));
//                pdst[dstIdx] += psrc[srcIdx] * (fwei * fscale);
//            }
//        }
//    });
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
