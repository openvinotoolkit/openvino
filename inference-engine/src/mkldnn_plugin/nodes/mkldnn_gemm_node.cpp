// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_gemm_node.h"
#include <legacy/ie_layers.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNGemmNode::MKLDNNGemmNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNGemmNode::getSupportedDescriptors() {
    auto* gemmLayer = dynamic_cast<GemmLayer*>(getCnnLayer().get());

    if (gemmLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert gemm layer.";

    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto inDims0 = getParentEdgeAt(0)->getDims();
    auto inDims1 = getParentEdgeAt(1)->getDims();
    auto outDims = getChildEdgeAt(0)->getDims();

    alpha = gemmLayer->alpha;
    beta = gemmLayer->beta;
    transposeA = gemmLayer->transpose_a;
    transposeB = gemmLayer->transpose_b;

    if ((inDims0.ndims() < 2 || inDims0.ndims() > 4) ||
        (inDims1.ndims() < 2 || inDims1.ndims() > 4))
        THROW_IE_EXCEPTION << "Unsupported input dims count for layer " << getName();

    if (outDims.ndims() < 2 || outDims.ndims() > 4)
        THROW_IE_EXCEPTION << "Unsupported output dims count for layer " << getName();

    if (inDims0.ndims() != inDims1.ndims() || inDims0.ndims() != outDims.ndims())
        THROW_IE_EXCEPTION << "Invalid dims count for layer " << getName();

    int nDims = inDims0.ndims();
    xAxis = nDims - 1;
    yAxis = nDims - 2;
    auto xAxis0 = transposeA ? yAxis : xAxis;
    auto yAxis0 = transposeA ? xAxis : yAxis;
    auto xAxis1 = transposeB ? yAxis : xAxis;
    auto yAxis1 = transposeB ? xAxis : yAxis;

    // The check inDims0[xAxis] != inDims1[yAxis] is correct due to layer semantic
    // coverity[copy_paste_error]
    if (inDims0[xAxis0] != inDims1[yAxis1] || inDims0[yAxis0] != outDims[yAxis] || inDims1[xAxis1] != outDims[xAxis])
        THROW_IE_EXCEPTION << "Spatial input and output dimensions are incorrect for layer " << getName();

    isThreeInputs = getParentEdges().size() == 3;

    if (isThreeInputs) {
        auto inDims2 = getParentEdgeAt(2)->getDims();

        if (inDims2.ndims() < 2 || inDims2.ndims() > 4)
            THROW_IE_EXCEPTION << "Unsupported output dims count for layer " << getName();

        if (inDims2.ndims() != outDims.ndims())
            THROW_IE_EXCEPTION << "Invalid dims count for layer " << getName();

        if (inDims2[yAxis] != outDims[yAxis] || inDims2[xAxis] != outDims[xAxis])
            THROW_IE_EXCEPTION << "Spatial input and output dimensions are incorrect for layer " << getName();
    }

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if (isThreeInputs) {
            auto inDims2 = getParentEdgeAt(2)->getDims();

            if (inDims2[dim_idx] != outDims[dim_idx] && inDims2[dim_idx] != 1)
                THROW_IE_EXCEPTION << "Input batch dimensions are incorrect for layer " << getName();

            int cOffset = 1;
            for (int i = dim_idx + 1; i < nDims; i++)
                cOffset *= inDims2[i];
            cOffsets.push_back(inDims2[dim_idx] == outDims[dim_idx] ? cOffset : 0);
        }

        if ((inDims0[dim_idx] != outDims[dim_idx] && inDims0[dim_idx] != 1) ||
            (inDims1[dim_idx] != outDims[dim_idx] && inDims1[dim_idx] != 1)) {
            THROW_IE_EXCEPTION << "Input batch dimensions are incorrect for layer " << getName();
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

void MKLDNNGemmNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto inPrec0 = getCnnLayer()->insData[0].lock()->getPrecision();
    auto inPrec1 = getCnnLayer()->insData[1].lock()->getPrecision();
    if ((inPrec0 != Precision::U8 && inPrec0 != Precision::I8) || inPrec1 != Precision::I8 || isThreeInputs) {
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

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;

    auto createDataConfig = [](const MKLDNNDims& dims, memory::data_type dataType) -> InferenceEngine::DataConfig {
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = MKLDNNMemoryDesc(dims, dataType, MKLDNNMemory::GetPlainFormat(dims));
        return dataConfig;
    };

    config.inConfs.push_back(createDataConfig(getParentEdgeAt(0)->getDims(), inputDataType0));
    config.inConfs.push_back(createDataConfig(getParentEdgeAt(1)->getDims(), inputDataType1));
    if (isThreeInputs) {
        auto inputDataType2 = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);
        config.inConfs.push_back(createDataConfig(getParentEdgeAt(2)->getDims(), inputDataType2));
    }

    config.outConfs.push_back(createDataConfig(getChildEdgeAt(0)->getDims(), outputDataType));

    supportedPrimitiveDescriptors.push_back(PrimitiveDescInfo(config, impl_desc_type::gemm_any, MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims())));
}

void MKLDNNGemmNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    MKLDNNNode::initOptimalPrimitiveDescriptor();

    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
}

void MKLDNNGemmNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory isn't allocated.";
    if (!src0MemPtr || !src0MemPtr->GetPrimitivePtr() || !src1MemPtr || !src1MemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory isn't allocated.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor isn't set.";

    if (isThreeInputs) {
        auto& src2MemPtr = getParentEdgeAt(2)->getMemoryPtr();
        if (!src2MemPtr || !src2MemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Input memory isn't allocated.";
    }
}

inline void process_gemm(char transa, char transb, int M, int N, int K, float alpha, const float *A, int lda,
                         const float *B, int ldb, float beta, float *C, int ldc) {
    mkldnn_sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void process_gemm(char transa, char transb, int M, int N, int K, float alpha, const uint16_t *A, int lda,
                         const uint16_t *B, int ldb, float beta, float *C, int ldc) {
    mkldnn_gemm_bf16bf16f32(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
void MKLDNNGemmNode::process_data() {
    auto inDims0 = getParentEdgeAt(0)->getDims();
    auto inDims1 = getParentEdgeAt(1)->getDims();
    auto outDims = getChildEdgeAt(0)->getDims();

    auto& srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto& srcMemory1 = getParentEdgeAt(1)->getMemory();

    const T0 *src0_ptr = reinterpret_cast<const T0*>(srcMemory0.GetData()) +
                              srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
    const T1 *src1_ptr = reinterpret_cast<const T1*>(srcMemory1.GetData()) +
                             srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_ptr = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
                     getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    int MB1 = outDims.ndims() == 4 ? batchToProcess() : 1;
    int MB2 = outDims.ndims() == 3 ? batchToProcess() : outDims.ndims() > 3 ? outDims[outDims.ndims() - 3] : 1;
    int M = outDims[yAxis];
    int N = outDims[xAxis];
    int K = transposeA ? inDims0[yAxis] : inDims0[xAxis];

    const char transa = transposeA ? 'T' : 'N';
    const char transb = transposeB ? 'T' : 'N';

    int lda = transposeA ? M : K;
    int ldb = transposeB ? K : N;
    int ldc = N;

    const float *src2_ptr;
    if (isThreeInputs) {
        auto& srcMemory2 = getParentEdgeAt(2)->getMemory();
        src2_ptr = reinterpret_cast<const float *>(srcMemory2.GetData()) +
                   srcMemory2.GetDescriptor().data.layout_desc.blocking.offset_padding;
    } else {
        src2_ptr = dst_ptr;
    }

    if (!isThreeInputs) {
        beta = 0.f;
    }

    for (int b1 = 0; b1 < MB1; b1++) {
        const T0 *a_ptr = src0_ptr;
        const T1 *b_ptr = src1_ptr;
        const float *c_ptr = src2_ptr;
        float *d_ptr = dst_ptr;

        for (int b2 = 0; b2 < MB2; b2++) {
            if (isThreeInputs) {
                cpu_memcpy(d_ptr, c_ptr, M * N * sizeof(float));
                c_ptr += cOffsets[0];
            }

            process_gemm(transa, transb, M, N, K, alpha, a_ptr, lda, b_ptr, ldb, beta, d_ptr, ldc);

            a_ptr += aOffsets[0];
            b_ptr += bOffsets[0];
            d_ptr += M * N;
        }

        src0_ptr += aOffsets[1];
        src1_ptr += bOffsets[1];
        dst_ptr += MB2 * M * N;

        if (isThreeInputs) {
            src2_ptr += cOffsets[1];
        }
    }
}

void MKLDNNGemmNode::execute(mkldnn::stream strm) {
    switch (getParentEdgeAt(0)->getDesc().getPrecision()) {
        case Precision::FP32:
            process_data<float, float>();
            break;
        case Precision::BF16:
            process_data<uint16_t, uint16_t>();
            break;
        case Precision::I8:
            process_data<int8_t, int8_t>();
            break;
        case Precision::U8:
            process_data<uint8_t, int8_t>();
            break;
        default:
            THROW_IE_EXCEPTION << "Gemm node: first input has unsupported precision";
    }
}

bool MKLDNNGemmNode::created() const {
    return getType() == Gemm;
}

int MKLDNNGemmNode::getMaxBatch() {
    if (!outDims.empty())
        return outDims[0][0];
    return 0;
}
REG_MKLDNN_PRIM_FOR(MKLDNNGemmNode, Gemm);
