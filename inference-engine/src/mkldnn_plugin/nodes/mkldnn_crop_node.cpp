// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_crop_node.h"
#include <ie_layers.h>
#include <string>
#include <algorithm>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNCropNode::MKLDNNCropNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

void MKLDNNCropNode::getSupportedDescriptors() {
    CropLayer* cropLayer = dynamic_cast<CropLayer*>(getCnnLayer().get());

    if (cropLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert crop layer.";

    channelAxis = 1;
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2) {
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    }

    MKLDNNDims childDims = getChildEdgeAt(0)->getDims();

    offsets.resize(static_cast<size_t>(childDims.ndims()));  // plus one dim for batch
    dims.resize(static_cast<size_t>(childDims.ndims()));  // plus one dim for batch
    for (int i = 0; i < childDims.ndims(); i++)
        dims[i] = childDims[i];

    for (int i = 0; i < cropLayer->axis.size(); i++) {
        offsets[cropLayer->axis[i]] = cropLayer->offset[i];
    }

    if (cropLayer->axis.size() == dims.size()) {
        for (size_t i = 0; i < cropLayer->axis.size(); i++) {
            if (cropLayer->axis[i] == 1) {
                channelAxis = static_cast<int>(i);
                break;
            }
        }
    }

    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNCropNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto& inDims = getParentEdgeAt(0)->getDims();
    if (inDims.ndims() != 4) {
        THROW_IE_EXCEPTION << "Crop supports only 4d blobs.";
    }

    memory::format fmt = memory::format::nchw;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(getParentEdges().size());
    config.outConfs.resize(1);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace = -1;
        config.inConfs[i].constant = i != 0;
        config.inConfs[i].desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, fmt);
    }
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);

    if (channelAxis >= 0 && dims[channelAxis] % 8 == 0) {
        fmt = memory::format::nChw8c;
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, fmt);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        if (dims[channelAxis] % 16 == 0) {
            fmt = memory::format::nChw16c;
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, fmt);
            config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        }
    }
}

void MKLDNNCropNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNCropNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();

    int m_block_size = 1;
    if (!MKLDNNMemory::IsPlainFormat(parentMem.GetFormat())) {
        m_block_size = parentMem.GetDescriptor().data.layout_desc.blocking.block_dims[1];
    }
    int m_inner_dim = dims[dims.size() - 1] * m_block_size;

    const memory &dst_d = getChildEdgeAt(0)->getMemory().GetPrimitive();

    int dst_ndims = dst_d.get_primitive_desc().desc().data.ndims;

    // TODO: Rewrite it in general case. For every tensor
    // and rank, without using letter N,C,H,W
    int OFFSET_N = (dst_ndims > 0) ? offsets[0] : 0;
    int OFFSET_C = (dst_ndims > 1) ? offsets[1] : 0;
    int OFFSET_H = (dst_ndims > 2) ? offsets[2] : 0;
    int OFFSET_W = (dst_ndims > 3) ? offsets[3] : 0;

    // TODO: Check applicability of dyn_batch_lim in early steps.
    //       crop of batch dimension doesn't support dyn batch.
    const int ON = (dst_ndims  > 0) ? std::min<int>(batchToProcess(), getChildEdgeAt(0)->getDims()[0]) : 1;
    const int OC = (dst_ndims  > 1) ? dims[1] : 1;
    const int OH = (dst_ndims  > 2) ? dims[2] : 1;
    const int OW = (dst_ndims  > 3) ? dims[3] : 1;

    memory::dims src_dims = parentMem.GetDims();
    int src_ndims = static_cast<int>(src_dims.size());

    const int IC = (src_ndims  > 1) ? rnd_up(src_dims[1], m_block_size) : 1;
    const int IH = (src_ndims  > 2) ? src_dims[2] : 1;
    const int IW = (src_ndims  > 3) ? src_dims[3] : 1;

    const auto *src_data = reinterpret_cast<const float*>(parentMem.GetData()) +
            parentMem.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_data = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

#ifdef _WIN32
    for (int n = 0; n < ON; ++n) {
        for (int c = 0; c < OC; c += m_block_size) {
            for (int h = 0; h < OH; ++h) {
                int dst_ind =
                        n*OC*OH*OW + c*OH*OW +
                        h*OW*m_block_size;

                int src_ind =
                        (n+OFFSET_N)*IC*IH*IW +
                        (c+OFFSET_C)*IH*IW +
                        (h+OFFSET_H)*IW*m_block_size +
                        OFFSET_W*m_block_size;

                memcpy(dst_data + dst_ind, src_data + src_ind, m_inner_dim * sizeof(float));
            }
        }
    }
#else
    parallel_for2d(ON, (OC / m_block_size), [&](int n, int c) {
        int dst_ind = (n*OC + c*m_block_size)*OH*OW;

        int src_ind = ((n+OFFSET_N)*IC + (c*m_block_size+OFFSET_C))*IH*IW +
                      (OFFSET_H*IW + OFFSET_W)*m_block_size;

        for (int h = 0; h < OH; ++h) {
            memcpy(dst_data + dst_ind, src_data + src_ind, m_inner_dim * sizeof(float));

            src_ind += IW*m_block_size;
            dst_ind += OW*m_block_size;
        }
    });
#endif
}

bool MKLDNNCropNode::created() const {
    return getType() == Crop;
}
