// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlas_gemm.hpp"

#include <cstdint>
#include <memory>

#include "cpu_memory.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "mlas/sgemm.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mlas/mlas_gemm.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

using namespace executor;
using namespace dnnl;
using namespace ov::element;

static MemoryPtr prepareWeightMemory(const MemoryPtr weightsMemory,
                                     const ExecutorContext::CPtr context,
                                     const bool weightsTransposed) {
    DEBUG_LOG("MlasGemmExecutor: prepack weights");
    const auto& wgtDims = weightsMemory->getStaticDims();
    // Weights are transposed by MatMulConstTransposesExtraction
    // K is the IC of weight
    // the weight is reshaped to [-1, K] in ConvertMatMulToFC
    const auto K = wgtDims[1];
    const auto N = wgtDims[0];

    auto packedBsize = mlas_sgemm_pack_get_size(N, K);

    auto create = [&]() {
        float* weightPtr = weightsMemory->getDataAs<float>();
        size_t ldb = weightsTransposed ? K : N;
        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine(),
                                                  intel_cpu::CpuBlockedMemoryDesc(i8, intel_cpu::Shape{packedBsize}));
        float* prepackedDst = _ptr->getDataAs<float>();
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
        return *weightCache->findOrCreate(string_hash, create);
    }

    DEBUG_LOG("MlasGemmExecutor: Weights cache is not available");
    return create();
}

// @todo use VERIFY macro for the checks
bool MlasGemmExecutor::supports(const FCConfig& config) {
    if (!config.postOps.empty()) {
        DEBUG_LOG("MlasGemmExecutor: PostOps are not supported");
        return false;
    }
    const auto& weiDesc = config.descs.at(ARG_WEI);
    const auto& dstDesc = config.descs.at(ARG_DST);

    // MLAS cannot support weight dims > 2, e.g. [1,64,9,9] * [10,64,9,9]
    const auto& weightsDims = weiDesc->getShape().getStaticDims();
    if (weightsDims.size() > 2) {
        if (!std::all_of(weightsDims.begin() + 2, weightsDims.end(), [](const Dim dim) {
                return dim == 1;
            })) {
            DEBUG_LOG("MlasGemmExecutor: weights dims > 2 are not supported");
            return false;
        }
    }

    if (config.attrs.withBias) {
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

MlasGemmExecutor::MlasGemmExecutor(const FCAttrs& attrs,
                                   const PostOps& postOps,
                                   const MemoryArgs& memory,
                                   const ExecutorContext::CPtr context)
    : m_attrs(attrs),
      m_memoryArgs(memory),
      packedWeights(prepareWeightMemory(memory.at(ARG_WEI), context, !attrs.weightsNonTransposed)) {}

bool MlasGemmExecutor::update(const MemoryArgs& memory) {
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();
    const auto& wgtDims = weiDesc->getShape().getStaticDims();
    // Weights are transposed by MatMulConstTransposesExtraction
    // K is the IC of weight
    // the weight is reshaped to [-1, K] in ConvertMatMulToFC
    K = wgtDims[1];
    N = wgtDims[0];

    const auto& outDims = dstDesc->getShape().getStaticDims();
    if (outDims.size() > 2) {
        M = std::accumulate(outDims.begin(), outDims.end() - 1, 1, std::multiplies<size_t>());
    } else {
        M = outDims[0];
    }
    return true;
}

void MlasGemmExecutor::execute(const MemoryArgs& memory) {
    const auto srcRawMemPtr = memory.at(ARG_SRC)->getDataAs<float>();
    const auto weiRawMemPtr = packedWeights->getDataAs<float>();
    const auto dstRawMemPtr = memory.at(ARG_DST)->getDataAs<float>();
    const auto biasRawMemPtr = memory.at(ARG_BIAS)->getDataAs<float>();

    const auto lda = K;
    const auto ldb = K;
    const auto ldc = N;

    mlas_sgemm_compute("N",
                       "N",
                       M,
                       N,
                       K,
                       1.0f,
                       srcRawMemPtr,
                       lda,
                       weiRawMemPtr,
                       ldb,
                       0.0f,
                       dstRawMemPtr,
                       ldc,
                       biasRawMemPtr);
}

void MlasGemmExecutor::moveMemToNumaNode(int numaNodeID) {
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
