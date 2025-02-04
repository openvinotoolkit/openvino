// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kleidiai_mm.hpp"

#include <cstdint>
#include <memory>
//#include <iomanip>

#include "cpu_memory.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "utils/cpu_utils.hpp"

#include "openvino/core/parallel.hpp"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN -3.4028235e38f

namespace ov {
namespace intel_cpu {

using namespace executor;
using namespace ov::element;

template <typename T>
static std::vector<T> normalizeDimsTo2D(const std::vector<T>& dims) {
    return {std::accumulate(dims.begin(), dims.end() - 1, (T)1, std::multiplies<T>()), dims[dims.size() - 1]};
}

bool MatMulKleidiAIExecutor::supports(const FCConfig& config) {
    //if (!config.attrs.weightsNonTransposed)
    //    return false;
    return true;
}

MatMulKleidiAIExecutor::MatMulKleidiAIExecutor(const FCAttrs& attrs,
                                               const PostOps& postOps,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context)
    : m_attrs(attrs),
      m_memoryArgs(memory) {
    aclfcAttrs.inputPrecision = memory.at(ARG_SRC)->getDescPtr()->getPrecision();
    aclfcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;
    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
    auto originalWeightsDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& wgtDims = originalWeightsDesc->getShape().getStaticDims();
    const VectorDims wgtDims2D = reshapeDownToRank<2>(wgtDims);
    originalWeightsDesc = std::make_shared<CpuBlockedMemoryDesc>(originalWeightsDesc->getPrecision(), Shape{wgtDims2D});
    auto dnnlSrcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(originalWeightsDesc);
    auto dstDesc = originalWeightsDesc->cloneWithNewPrecision(aclfcAttrs.inputPrecision);
    auto dnnlDstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc);
    if (!aclfcAttrs.weightsNonTransposed) {
        dnnlDstDesc = acl_fc_executor::makeTransposedWeightDescriptor(dnnlDstDesc, dnnlSrcDesc);
        aclfcAttrs.isWeightsRepacked = true;
    }
    packedWeights = acl_fc_executor::reorderWeights(memory, context, aclfcAttrs, dnnlSrcDesc, dnnlDstDesc);
}

bool MatMulKleidiAIExecutor::update(const MemoryArgs& memory) {
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

/*void printMatrix(float* matrix, int rows, int cols) {
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void multiplyMatrix(float* src, float* wei, float* dst, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dst[i * N + j] = 0;
            for (int k = 0; k < K; k++) {
                dst[i * N + j] += src[i * K + k] * wei[k * N + j];
            }
        }
    }
}

void transposeMatrix(float* ptr, int M, int N, float* transposed) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            transposed[col * M + row] = ptr[row * N + col];
        }
    }
}*/

void MatMulKleidiAIExecutor::execute(const MemoryArgs& memory) {
    size_t BLOCK_SIZE = 8;
    auto srcMem = memory.at(ARG_SRC);
    auto weiMem = memory.at(ARG_WEI);
    auto dstMem = memory.at(ARG_DST);
    auto biasMem = memory.at(ARG_BIAS);
    auto srcDims = normalizeDimsTo2D(srcMem->getDesc().getShape().getDims());
    auto weiDims = weiMem->getDesc().getShape().getDims();
    auto M = srcDims[0];
    auto K = srcDims[1];
    auto N = weiDims[0];

    const size_t lhs_stride = K * sizeof(float);
    const size_t rhs_stride = N * sizeof(float);
    const size_t dst_stride_row = N * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
    float* rhs_packed = new float[rhs_packed_size];

    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();

    float* lhs = srcMem->getDataAs<float>();
    float* rhs = static_cast<float*>(packedWeights->getData());
    float* dst = dstMem->getDataAs<float>();
    float* bias = biasMem->getDataAs<float>();

    /*std::cout << "SRC:\n";
    printMatrix(lhs, 4, 2);
    std::cout << "WEI:\n";
    printMatrix(rhs, 3, 2);
    std::cout << "PACKED WEI:\n";
    printMatrix(static_cast<float*>(packedWeights->getData()), 2, 3);

    float transposed[6] = {0};
    transposeMatrix(rhs, 3, 2, &transposed[0]);
    std::cout << "TRANSPOSED WEI:\n";
    printMatrix(&transposed[0], 2, 3);

    float testDst[12] = {0};
    multiplyMatrix(lhs, rhs, testDst, 4, 2, 3);
    std::cout << "DST (SRC * WEI):\n";
    printMatrix(&testDst[0], 4, 3);

    multiplyMatrix(lhs, transposed, testDst, 4, 2, 3);
    std::cout << "DST (SRC * TRANSPOSED WEI):\n";
    printMatrix(&testDst[0], 4, 3);*/

    if (bias == nullptr) {
        bias = new float[N];
        memset(bias, 0, N * sizeof(float));
    }

    kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(
            1, N, K, nr, kr, sr,  // Packing arguments
            rhs_stride,           // RHS stride
            rhs,                  // RHS
            bias,                 // Bias
            nullptr,              // Scale
            rhs_packed,           // RHS packed
            0, nullptr);

    size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    parallel_for(n_blocks, [&](size_t n_block) {
        size_t n_start = (n_block * BLOCK_SIZE);
        size_t n_end = std::min(n_start + BLOCK_SIZE, N);
        size_t n_block_size = n_end - n_start;
        const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(n_start, K);
        const size_t dst_offset = ukernel.get_dst_offset(0, n_start, dst_stride_row);
        const float* rhs_ptr = (rhs_packed + rhs_packed_offset / sizeof(float));
        float* dst_ptr = (dst + dst_offset / (sizeof(float)));
        ukernel.run_matmul(
                M,
                n_block_size,
                K,
                lhs,
                lhs_stride,
                rhs_ptr,
                dst_ptr,
                dst_stride_row,
                dst_stride_col,
                FLOAT_MIN,
                FLOAT_MAX);
    });

    //std::cout << "KLEIDIAI DST:\n";
    //printMatrix(dst, 4, 3);
}

void MatMulKleidiAIExecutor::moveMemToNumaNode(int numaNodeID) {
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