// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_roi_pooling_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNROIPoolingNode::MKLDNNROIPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket) : MKLDNNNode(layer, eng, socket) {}

void MKLDNNROIPoolingNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    GenericLayer* genericLayer = getCnnLayer().get();

    if (genericLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert ROIPooling layer.";

    if (getParentEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    pooled_h = genericLayer->GetParamAsInt("pooled_h");
    pooled_w = genericLayer->GetParamAsInt("pooled_w");
    spatial_scale = genericLayer->GetParamAsFloat("spatial_scale");
    std::string m = genericLayer->GetParamAsString("method", "max");
    if (m == "max") {
        method = mkldnn::algorithm::roi_pooling_max;
    } else if (m == "bilinear") {
        method = mkldnn::algorithm::roi_pooling_bilinear;
    } else {
        THROW_IE_EXCEPTION << "Unsupported roi pooling method";
    }

    auto parentDims = getParentEdgeAt(0)->getDims();
    for (auto format : getAvailableFormatsForDims(parentDims)) {
        std::vector<InferenceEngine::TensorDesc> srcs;
        srcs.push_back(MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, format));
        srcs.push_back(MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), inputDataType, memory::nc));
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, format);

        createDescriptor(srcs, {out_candidate});
    }
}

void MKLDNNROIPoolingNode::createPrimitive() {
    if (prim)
        return;

    std::vector<memory::desc> srcs;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        srcs.push_back(getParentEdgeAt(i)->getMemory().GetDescriptor());
    }

    memory::desc out_candidate = getChildEdgeAt(0)->getMemory().GetDescriptor();

    MKLDNNDescriptor desc(std::shared_ptr<roi_pooling_forward::desc>(
            new roi_pooling_forward::desc(prop_kind::forward_scoring, method, srcs, out_candidate, pooled_h, pooled_w,
                                          spatial_scale)));

    descs[0] = desc;
    std::shared_ptr<roi_pooling_forward::desc> selected_desc_ptr = descs[0];

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set for node " << getName() << ".";

    auto prim_desc = roi_pooling_forward::primitive_desc(*selected_desc_ptr, getEngine());
    primitive_desc_iterator itpd = descs[0].createPrimitiveDescriptorIterator(getEngine());

    std::vector<primitive::at> src_p;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        src_p.push_back(getParentEdgeAt(i)->getMemoryPtr()->GetPrimitive());
    }
    prim.reset(new roi_pooling_forward(prim_desc, src_p, getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNROIPoolingNode::created() const {
    return getType() == ROIPooling;
}

void MKLDNNROIPoolingNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                            const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    std::vector<memory::desc> srcs;
    srcs.push_back(MKLDNNMemoryDesc(inputDesc[0]));
    srcs.push_back(MKLDNNMemoryDesc(inputDesc[1]));
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);

    MKLDNNDescriptor desc(std::shared_ptr<roi_pooling_forward::desc>(
            new roi_pooling_forward::desc(prop_kind::forward_scoring, method, srcs, out_candidate, pooled_h, pooled_w,
                                          spatial_scale)));
    descs.push_back(desc);
}
REG_MKLDNN_PRIM_FOR(MKLDNNROIPoolingNode, RoiPooling);
