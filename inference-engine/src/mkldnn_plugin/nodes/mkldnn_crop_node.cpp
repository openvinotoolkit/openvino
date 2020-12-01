// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_crop_node.h"
#include <legacy/ie_layers.h>
#include <string>
#include <algorithm>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNCropNode::MKLDNNCropNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

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
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    if (inputDataType != outputDataType) {
        outputDataType = inputDataType; // Crop doesn't convert precisions, only moves data
    }

    auto& inDims = getParentEdgeAt(0)->getDims();
    if (inDims.ndims() != 2 && inDims.ndims() != 4 && inDims.ndims() != 5) {
        THROW_IE_EXCEPTION << "Crop supports only 2d, 4d and 5d blobs.";
    }

    memory::format fmt = memory::format::format_undef;
    switch (inDims.ndims()) {
        case 2: fmt = memory::format::nc; break;
        case 4: fmt = memory::format::nchw; break;
        case 5: fmt = memory::format::ncdhw; break;
    }

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

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, fmt);

    if ((inDims.ndims() == 4 || inDims.ndims() == 5) && channelAxis >= 0 && dims[channelAxis] % 8 == 0) {
        fmt = inDims.ndims() == 5 ? memory::format::nCdhw8c : memory::format::nChw8c;
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, fmt);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, fmt);
        if (dims[channelAxis] % 16 == 0) {
            fmt = inDims.ndims() == 5 ? memory::format::nCdhw16c : memory::format::nChw16c;
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, fmt);
            config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, fmt);
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
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
}

void MKLDNNCropNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();

    int m_block_size = 1;
    if (!MKLDNNMemory::IsPlainFormat(parentMem.GetFormat())) {
        m_block_size = parentMem.GetDescriptor().data.layout_desc.blocking.block_dims[1];
    }
    const int m_inner_dim = dims[dims.size() - 1] * m_block_size;

    const memory &dst_d = getChildEdgeAt(0)->getMemory().GetPrimitive();

    const int dst_ndims = dst_d.get_primitive_desc().desc().data.ndims;

    // TODO: Rewrite it in general case. For every tensor
    // and rank, without using letter N,C,D,H,W
    const int OFFSET_N = (dst_ndims > 0) ? offsets[0] : 0;
    const int OFFSET_C = (dst_ndims > 1) ? offsets[1] : 0;
    const int OFFSET_D = (dst_ndims > 4) ? offsets[offsets.size() - 3] : 0;
    const int OFFSET_H = (dst_ndims > 2) ? offsets[offsets.size() - 2] : 0;
    const int OFFSET_W = (dst_ndims > 3) ? offsets[offsets.size() - 1] : 0;

    // TODO: Check applicability of dyn_batch_lim in early steps.
    //       crop of batch dimension doesn't support dyn batch.
    const int ON = (dst_ndims  > 0) ? std::min<int>(batchToProcess(), getChildEdgeAt(0)->getDims()[0]) : 1;
    const int OC = (dst_ndims  > 1) ? dims[1] : 1;
    const int OD = (dst_ndims  > 4) ? dims[dims.size() - 3] : 1;
    const int OH = (dst_ndims  > 2) ? dims[dims.size() - 2] : 1;
    const int OW = (dst_ndims  > 3) ? dims[dims.size() - 1] : 1;

    memory::dims src_dims = parentMem.GetDims();
    int src_ndims = static_cast<int>(src_dims.size());

    const int IC = (src_ndims  > 1) ? rnd_up(src_dims[1], m_block_size) : 1;
    const int ID = (src_ndims  > 4) ? src_dims[src_dims.size() - 3] : 1;
    const int IH = (src_ndims  > 2) ? src_dims[src_dims.size() - 2] : 1;
    const int IW = (src_ndims  > 3) ? src_dims[src_dims.size() - 1] : 1;

    const uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(parentMem.GetDataType()));

    const auto *src_data = reinterpret_cast<const uint8_t *>(parentMem.GetData()) +
            itemSize * parentMem.GetDescriptor().data.layout_desc.blocking.offset_padding;
    auto *dst_data = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemory().GetData()) +
            itemSize * getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    if (OD == 1 && OH == 1 && OW == 1 && ID == 1 && IH == 1 && IW == 1) {
        parallel_for(ON, [&](int n) {
            cpu_memcpy(dst_data + itemSize * n * OC, src_data + itemSize *((n+OFFSET_N)*IC + OFFSET_C), OC * itemSize);
        });
    } else {
        parallel_for2d(ON, (OC / m_block_size), [&](int n, int c) {
            for (int d = 0; d < OD; ++d) {
                int dst_ind = (n*OC + c*m_block_size)*OD*OH*OW + d*m_block_size*OH*OW;

                int src_ind = ((n+OFFSET_N)*IC + (c*m_block_size+OFFSET_C))*ID*IH*IW +
                              ((d+OFFSET_D)*IH*IW + OFFSET_H*IW + OFFSET_W)*m_block_size;

                for (int h = 0; h < OH; ++h) {
                    cpu_memcpy(dst_data + itemSize * dst_ind, src_data + itemSize * src_ind, m_inner_dim * itemSize);

                    src_ind += IW * m_block_size;
                    dst_ind += OW * m_block_size;
                }
            }
        });
    }
}

bool MKLDNNCropNode::created() const {
    return getType() == Crop;
}
REG_MKLDNN_PRIM_FOR(MKLDNNCropNode, Crop);
