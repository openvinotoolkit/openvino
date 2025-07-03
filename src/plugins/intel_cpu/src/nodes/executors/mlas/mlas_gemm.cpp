// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlas_gemm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "mlas/sgemm.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mlas/mlas_gemm.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

using namespace executor;
using namespace dnnl;
using namespace ov::element;

static Dim batchDim(const VectorDims& dims) {
    return std::accumulate(dims.begin(), dims.end() - 1, 1, std::multiplies<>());
}

static MemoryPtr prepareWeightMemory(const MemoryPtr weightsMemory,
                                     const ExecutorContext::CPtr context,
                                     const bool weightsTransposed) {
    DEBUG_LOG("MlasGemmExecutor: prepack weights");
    const auto& wgtDims = weightsMemory->getStaticDims();
    // Weights are transposed by MatMulConstTransposesExtraction
    // K is the IC of weight
    // the weight is reshaped to [-1, K] in ConvertMatMulToFC
    Dim K = wgtDims.back();
    Dim N = batchDim(wgtDims);

    auto packedBsize = mlas_sgemm_pack_get_size(N, K);

    auto create = [&]() {
        auto* weightPtr = weightsMemory->getDataAs<float>();
        size_t ldb = weightsTransposed ? K : N;

        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine(),
                                                  intel_cpu::CpuBlockedMemoryDesc(i8, intel_cpu::Shape{packedBsize}));
        auto* prepackedDst = _ptr->getDataAs<float>();
        DEBUG_LOG("MlasGemmExecutor: cache miss, perform packing");
        mlas_sgemm_pack(weightsTransposed ? "T" : "F", N, K, ldb, weightPtr, prepackedDst);
        return _ptr;
    };

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        std::string format = "gemm_mlas_" + std::to_string(N) + "_" + std::to_string(K);
        const std::string string_hash = format + "_" + std::to_string(weightsMemory->getSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(weightsMemory->getData()));
        DEBUG_LOG("MlasGemmExecutor: findOrCreate, string_hash: ", string_hash);
        return MemoryPtr(*weightCache->findOrCreate(string_hash, create));
    }

    DEBUG_LOG("MlasGemmExecutor: Weights cache is not available");
    return create();
}

// @todo use VERIFY macro for the checks
bool MlasGemmExecutor::supports(const FCConfig& config) {
    if (!config.attrs.postOps.empty()) {
        DEBUG_LOG("MlasGemmExecutor: PostOps are not supported");
        return false;
    }

    const auto& dstDesc = config.descs.at(ARG_DST);

    if (!config.descs.at(ARG_BIAS)->empty()) {
        const auto& biaDesc = config.descs.at(ARG_BIAS);
        const auto& biasDims = biaDesc->getShape().getStaticDims();
        const auto& outDims = dstDesc->getShape().getDims();
        const bool isByChannel = biasDims.back() == outDims.back();

        if (!isByChannel) {
            DEBUG_LOG("MlasGemmExecutor: only 'by channel' bias is supported");
            return false;
        }

        if (!std::all_of(biasDims.begin(), biasDims.end() - 1, [](const Dim dim) {
                return dim == 1;
            })) {
            DEBUG_LOG("MlasGemmExecutor: only 'by channel' bias is supported");
            return false;
        }
    }

    return true;
}

MlasGemmExecutor::MlasGemmExecutor(const FCAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context)
    : m_attrs(attrs),
      m_memoryArgs(memory),
      packedWeights(prepareWeightMemory(memory.at(ARG_WEI), context, !attrs.weightsNonTransposed)),

      N(batchDim(memory.at(ARG_WEI)->getStaticDims())),
      K(memory.at(ARG_WEI)->getStaticDims().back()) {}

bool MlasGemmExecutor::update(const MemoryArgs& memory) {
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto& outDims = dstDesc->getShape().getStaticDims();
    M = outDims.size() > 2 ? batchDim(outDims) : outDims[0];

    return true;
}

void MlasGemmExecutor::execute(const MemoryArgs& memory) {
    auto* const srcRawMemPtr = memory.at(ARG_SRC)->getDataAs<float>();
    auto* const weiRawMemPtr = packedWeights->getDataAs<float>();
    auto* const dstRawMemPtr = memory.at(ARG_DST)->getDataAs<float>();
    auto* const biasRawMemPtr = memory.at(ARG_BIAS)->getDataAs<float>();

    const auto lda = K;
    const auto ldb = K;
    const auto ldc = N;

    mlas_sgemm_compute("N",
                       "N",
                       M,
                       N,
                       K,
                       1.0F,
                       srcRawMemPtr,
                       lda,
                       weiRawMemPtr,
                       ldb,
                       0.0F,
                       dstRawMemPtr,
                       ldc,
                       biasRawMemPtr);
}

void MlasGemmExecutor::moveMemToNumaNode(int numaNodeID) {
    if (curNumaNode == numaNodeID) {
        return;
    }
    curNumaNode = numaNodeID;
    mbind_move(packedWeights, numaNodeID);
    if (m_attrs.withBias) {
        mbind_move(m_memoryArgs.at(ARG_BIAS), numaNodeID);
    }
}

}  // namespace ov::intel_cpu
