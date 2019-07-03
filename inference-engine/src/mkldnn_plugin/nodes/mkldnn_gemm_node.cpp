// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_gemm_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNGemmNode::MKLDNNGemmNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

void MKLDNNGemmNode::getSupportedDescriptors() {
    auto* gemmLayer = dynamic_cast<GemmLayer*>(getCnnLayer().get());

    if (gemmLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert gemm layer.";

    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().size() != 1)
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

    // The check inDims0[xAxis] != inDims1[yAxis] is correct due to layer semantic
    // coverity[copy_paste_error]
    if (inDims0[xAxis] != inDims1[yAxis] || inDims0[yAxis] != outDims[yAxis] || inDims1[xAxis] != outDims[xAxis])
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

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);

    auto same = [&] (memory::format fmt) -> PrimitiveDescInfo {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, fmt);
            config.inConfs.push_back(dataConfig);
        }

        InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
            config.outConfs.push_back(dataConfig);
        return {config, impl_desc_type::gemm_any};
    };

    supportedPrimitiveDescriptors.push_back(same(memory::any));
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

void MKLDNNGemmNode::execute(mkldnn::stream strm) {
    auto inDims0 = getParentEdgeAt(0)->getDims();
    auto inDims1 = getParentEdgeAt(1)->getDims();
    auto outDims = getChildEdgeAt(0)->getDims();

    auto& srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto& srcMemory1 = getParentEdgeAt(1)->getMemory();
    const float *src0_ptr = reinterpret_cast<const float*>(srcMemory0.GetData()) +
                            srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
    const float *src1_ptr = reinterpret_cast<const float*>(srcMemory1.GetData()) +
                            srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_ptr = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
                     getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    int MB1 = outDims.ndims() == 4 ? batchToProcess() : 1;
    int MB2 = outDims.ndims() == 3 ? batchToProcess() : outDims.ndims() > 3 ? outDims[outDims.ndims() - 3] : 1;
    int M = inDims0[yAxis];
    int N = inDims1[xAxis];
    int K = inDims0[xAxis];

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
        const float *a_ptr = src0_ptr;
        const float *b_ptr = src1_ptr;
        const float *c_ptr = src2_ptr;
        float *d_ptr = dst_ptr;

        for (int b2 = 0; b2 < MB2; b2++) {
            if (isThreeInputs) {
                memcpy(d_ptr, c_ptr, M * N * sizeof(float));
                c_ptr += cOffsets[0];
            }

            mkldnn_sgemm(&transb, &transa, &N, &M, &K, &alpha, b_ptr, &ldb, a_ptr, &lda, &beta, d_ptr, &ldc);

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

bool MKLDNNGemmNode::created() const {
    return getType() == Gemm;
}

int MKLDNNGemmNode::getMaxBatch() {
    if (!outDims.empty())
        return outDims[0][0];
    return 0;
}
