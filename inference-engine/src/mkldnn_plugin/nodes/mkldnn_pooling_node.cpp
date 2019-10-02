// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_pooling_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPoolingNode::MKLDNNPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket)
        : MKLDNNNode(layer, eng, socket) {}

void MKLDNNPoolingNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto * poolingLayer = dynamic_cast<PoolingLayer*>(getCnnLayer().get());
    if (poolingLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert pooling layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    type = poolingLayer->_type;
    exclude_pad = poolingLayer->_exclude_pad;

    invertVectorCopyUtoI(poolingLayer->_stride, stride);
    invertVectorCopyUtoI(poolingLayer->_kernel, kernel);
    auto allPads = getPaddings(*poolingLayer);
    invertVectorCopyUtoI(allPads.begin, paddingL);
    invertVectorCopyUtoI(allPads.end, paddingR);

    auto parentDims = getParentEdgeAt(0)->getDims();
    auto childDims = getChildEdgeAt(0)->getDims();
    if ((parentDims.ndims() < 4) || (parentDims.ndims() > 5))
        THROW_IE_EXCEPTION << "Pooling layer. Unsupported mode. Only 4D and 5D blobs are supported as input.";

    for (int i = 0; i < paddingR.size(); i++) {
        int krn = kernel[i];
        int src = getParentEdgeAt(0)->getDims()[2 + i];
        int dst = getChildEdgeAt(0)->getDims()[2 + i];

        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }
    if (this->getCnnLayer()->precision == Precision::I8) {
        // i8 layers supports only nhwc layout
        MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, memory::format::nhwc};
        MKLDNNMemoryDesc out_candidate{childDims, outputDataType, memory::format::nhwc};
        createDescriptor({ in_candidate }, { out_candidate });
    } else if ((parentDims.ndims() == 4 || parentDims.ndims() == 5) && parentDims[1] == 1) {
        inputDataType = memory::f32;
        outputDataType = memory::f32;
        // WA. We should force planar layout since it provides better performance
        MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, parentDims.ndims() == 5 ? memory::format::ncdhw : memory::format::nchw};
        MKLDNNMemoryDesc out_candidate{childDims, outputDataType, parentDims.ndims() == 5 ? memory::format::ncdhw : memory::format::nchw};
        createDescriptor({ in_candidate }, { out_candidate });
    } else {
        inputDataType = memory::f32;
        outputDataType = memory::f32;
        // It doesn't support any format
        for (auto format : getAvailableFormatsForDims(parentDims)) {
            MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, format};
            MKLDNNMemoryDesc out_candidate{childDims, outputDataType, format};
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void MKLDNNPoolingNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<pooling_forward::primitive_desc, pooling_forward::desc>();

    prim.reset(new pooling_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                   getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNPoolingNode::created() const {
    return getType() == Pooling;
}

void MKLDNNPoolingNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                         const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);

    algorithm alg;
    if (type == PoolingLayer::PoolType::AVG) {
        bool not_zero_l = false;
        for (auto lr : paddingL) {
            if (lr) {
                not_zero_l = true;
                break;
            }
        }
        if (!exclude_pad && not_zero_l)
            alg = pooling_avg_include_padding;
        else
            alg = pooling_avg_exclude_padding;
    } else if (type == PoolingLayer::PoolType::MAX) {
        alg = pooling_max;
    } else {
        // TODO: Handle rest of the possible: STOCH, ROI, SPACIAL_PYRAMID
        THROW_IE_EXCEPTION << "Unsupported pooling type";
    }

    std::shared_ptr<pooling_forward::desc> desc_ptr(
            new pooling_forward::desc(prop_kind::forward_scoring, alg,
                                      in_candidate, out_candidate,
                                      stride, kernel, paddingL, paddingR,
                                      mkldnn::padding_kind::zero));

    bool not_zero_r = false;
    for (auto pr : paddingR) {
        if (pr) {
            not_zero_r = true;
            break;
        }
    }
    if (alg == pooling_avg_include_padding && not_zero_r) {
        // In case of AVG including paddings the norm coeff should be calculated
        // with tacking into account original pads. So we need to restore
        // original values (R_padding = L_padding).
        //
        // WA. Because mkldnn uses different formula to calculate AVG norm coeff
        //     in compare with Caffe. In mkldnn coeff is always 1/(KH*KW)
        for (int i = 0; i < paddingL.size(); i++) desc_ptr->data.padding[1][i] = paddingL[i];
    }

    descs.emplace_back(desc_ptr);
}
REG_MKLDNN_PRIM_FOR(MKLDNNPoolingNode, Pooling);
