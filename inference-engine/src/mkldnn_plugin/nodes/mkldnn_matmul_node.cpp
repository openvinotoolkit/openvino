// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_matmul_node.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNMatMulNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        const auto shapeA = matMul->get_input_shape(0);
        const auto shapeB = matMul->get_input_shape(1);

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_shape(i).size();
            if (inShapeRank < 2 || inShapeRank > 4) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_shape().size();
        if (outShapeRank < 2 || outShapeRank > 4) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMatMulNode::MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Gemm node with name '" + getName() + "'";

        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        alpha = 1.f;
        beta = 0.f;
        transposeA = matMul->get_transpose_a();
        transposeB = matMul->get_transpose_b();
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNMatMulNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW()  << errorPrefix << " has incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW()  << errorPrefix << " has incorrect number of output edges for layer " << getName();

    auto inDims0 = getParentEdgeAt(0)->getShape().getStaticDims();
    auto inDims1 = getParentEdgeAt(1)->getShape().getStaticDims();
    auto outDims = getChildEdgeAt(0)->getShape().getStaticDims();

    if (inDims0.size() != inDims1.size() || inDims0.size() != outDims.size())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    int nDims = inDims0.size();
    xAxis = nDims - 1;
    yAxis = nDims - 2;
    auto xAxis0 = transposeA ? yAxis : xAxis;
    auto yAxis0 = transposeA ? xAxis : yAxis;
    auto xAxis1 = transposeB ? yAxis : xAxis;
    auto yAxis1 = transposeB ? xAxis : yAxis;

    // The check inDims0[xAxis] != inDims1[yAxis] is correct due to layer semantic
    // coverity[copy_paste_error]
    if (inDims0[xAxis0] != inDims1[yAxis1] || inDims0[yAxis0] != outDims[yAxis] || inDims1[xAxis1] != outDims[xAxis])
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((inDims0[dim_idx] != outDims[dim_idx] && inDims0[dim_idx] != 1) ||
            (inDims1[dim_idx] != outDims[dim_idx] && inDims1[dim_idx] != 1)) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }

        int aOffset = 1;
        for (int i = dim_idx + 1; i < nDims; i++)
            aOffset *= inDims0[i];
        aOffsets.push_back(inDims0[dim_idx] == outDims[dim_idx] ? aOffset : 0);

        int bOffset = 1;
        for (int i = dim_idx + 1; i < nDims; i++)
            bOffset *= inDims1[i];
        bOffsets.push_back(inDims1[dim_idx] == outDims[dim_idx] ? bOffset : 0);
    }

    for (unsigned long dim_idx = aOffsets.size(); dim_idx < 2; dim_idx++)
        aOffsets.push_back(0);
    for (unsigned long dim_idx = bOffsets.size(); dim_idx < 2; dim_idx++)
        bOffsets.push_back(0);
    for (unsigned long dim_idx = cOffsets.size(); dim_idx < 2; dim_idx++)
        cOffsets.push_back(0);
}

void MKLDNNMatMulNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto inPrec0 = getOriginalInputPrecisionAtPort(0);
    auto inPrec1 = getOriginalInputPrecisionAtPort(1);
    if ((inPrec0 != Precision::U8 && inPrec0 != Precision::I8) || inPrec1 != Precision::I8) {
        if (inPrec0 == Precision::BF16 || inPrec1 == Precision::BF16) {
            inPrec0 = Precision::BF16;
            inPrec1 = Precision::BF16;
        } else {
            inPrec0 = Precision::FP32;
            inPrec1 = Precision::FP32;
        }
    }

    auto inputDataType0 = MKLDNNExtensionUtils::IEPrecisionToDataType(inPrec0);
    auto inputDataType1 = MKLDNNExtensionUtils::IEPrecisionToDataType(inPrec1);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);

    NodeConfig config;
    config.dynBatchSupport = true;

    auto createDataConfig = [](const std::vector<size_t>& dims, memory::data_type dataType) -> PortConfig {
        PortConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(dims, dataType, MKLDNNMemory::GetPlainFormatByRank(dims.size()));
        return dataConfig;
    };

    config.inConfs.push_back(createDataConfig(getParentEdgeAt(0)->getShape().getStaticDims(), inputDataType0));
    config.inConfs.push_back(createDataConfig(getParentEdgeAt(1)->getShape().getStaticDims(), inputDataType1));
    config.outConfs.push_back(createDataConfig(getChildEdgeAt(0)->getShape().getStaticDims(), outputDataType));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::gemm_any);
}

void MKLDNNMatMulNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";
    auto config = selected_pd->getConfig();

     if (isConfigDefined(config))
         return;

    MKLDNNNode::initOptimalPrimitiveDescriptor();

    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
}

void MKLDNNMatMulNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate destination memory";
    if (!src0MemPtr || !src0MemPtr->GetPrimitivePtr() || !src1MemPtr || !src1MemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";

    auto inDims0 = src0MemPtr->GetDims();
    auto outDims = dstMemPtr->GetDims();

    params.src0_mem_ptr = src0MemPtr;
    params.src1_mem_ptr = src1MemPtr;
    params.dst_mem_ptr = dstMemPtr;

    params.ndims = outDims.size();

    params.MB1 = 1;
    params.MB2 = outDims.size() > 3 ? outDims[params.ndims - 3] : 1;

    params.M = outDims[yAxis];
    params.N = outDims[xAxis];
    params.K = transposeA ? inDims0[yAxis] : inDims0[xAxis];

    params.transa = transposeA ? 'T' : 'N';
    params.transb = transposeB ? 'T' : 'N';

    params.lda = transposeA ? params.M : params.K;
    params.ldb = transposeB ? params.K : params.N;
    params.ldc = params.N;

    params.shift1 = params.M * params.N * params.MB2;
    params.shift2 = params.M * params.N;

    runtimePrecision = getParentEdgeAt(0)->getMemory().GetDesc().getPrecision();
}

inline void process_gemm(char transa, char transb, int M, int N, int K, float alpha, const float *A, int lda,
                         const float *B, int ldb, float beta, float *C, int ldc) {
    mkldnn_sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void process_gemm(char transa, char transb, int M, int N, int K, float alpha, const uint16_t *A, int lda,
                         const uint16_t *B, int ldb, float beta, float *C, int ldc) {
    dnnl_gemm_bf16bf16f32(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void process_gemm(char transa, char transb, int M, int N, int K, float alpha, const uint8_t *A, int lda,
                         const int8_t *B, int ldb, float beta, float *C, int ldc) {
    const int32_t co = 0;
    int32_t *Ci = reinterpret_cast<int32_t *>(C);
    mkldnn_gemm_u8s8s32(transa, transb, 'F', M, N, K, alpha, A, lda, 0, B, ldb, 0, beta, Ci, ldc, &co);
    parallel_for(M * N, [&](size_t i) {
        C[i] = Ci[i];
    });
}

inline void process_gemm(char transa, char transb, int M, int N, int K, float alpha, const int8_t *A, int lda,
                         const int8_t *B, int ldb, float beta, float *C, int ldc) {
    const int32_t co = 0;
    int32_t *Ci = reinterpret_cast<int32_t *>(C);
    mkldnn_gemm_s8s8s32(transa, transb, 'F', M, N, K, alpha, A, lda, 0, B, ldb, 0, beta, Ci, ldc, &co);
    parallel_for(M * N, [&](size_t i) {
        C[i] = Ci[i];
    });
}

template<typename T0, typename T1>
inline void MKLDNNMatMulNode::process_data() {
    const T0* src0_ptr = reinterpret_cast<const T0*>(params.src0_mem_ptr->GetPtr());
    const T1* src1_ptr = reinterpret_cast<const T1*>(params.src1_mem_ptr->GetPtr());
    float* dst_ptr = reinterpret_cast<float*>(params.dst_mem_ptr->GetPtr());

    const int MB = batchToProcess();
    if (params.ndims == 4) {
        params.MB1 = MB;
    } else if (params.ndims == 3) {
        params.shift1 = params.shift1 * MB / params.MB2;
        params.MB2 = MB;
    }

    for (int b1 = 0; b1 < params.MB1; ++b1) {
        const T0 *a_ptr = src0_ptr;
        const T1 *b_ptr = src1_ptr;
        float *d_ptr = dst_ptr;

        for (int b2 = 0; b2 < params.MB2; ++b2) {
            process_gemm(params.transa, params.transb, params.M, params.N, params.K,
                         alpha, a_ptr, params.lda, b_ptr, params.ldb, beta, d_ptr, params.ldc);

            a_ptr += aOffsets[0];
            b_ptr += bOffsets[0];
            d_ptr += params.shift2;
        }

        src0_ptr += aOffsets[1];
        src1_ptr += bOffsets[1];
        dst_ptr += params.shift1;
    }
}

void MKLDNNMatMulNode::execute(mkldnn::stream strm) {
    switch (runtimePrecision) {
        case Precision::FP32: {
            process_data<float, float>();
            break;
        }
        case Precision::BF16: {
            process_data<uint16_t, uint16_t>();
            break;
        }
        case Precision::I8: {
            process_data<int8_t, int8_t>();
            break;
        }
        case Precision::U8: {
            process_data<uint8_t, int8_t>();
            break;
        }
        default:
            IE_THROW()  << errorPrefix << " has incorrect precision on first input";
    }
}

bool MKLDNNMatMulNode::created() const {
    return getType() == MatMul;
}

int MKLDNNMatMulNode::getMaxBatch() {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MKLDNNMatMulNode::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatMulNode, MatMul);
