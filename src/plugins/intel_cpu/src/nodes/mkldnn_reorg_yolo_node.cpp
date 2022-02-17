// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include <ngraph/opsets/opset2.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_reorg_yolo_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNReorgYoloNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto reorgYolo = std::dynamic_pointer_cast<const ngraph::opset2::ReorgYolo>(op);
        if (!reorgYolo) {
            errorMessage = "Only opset2 ReorgYolo operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNReorgYoloNode::MKLDNNReorgYoloNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = std::string(op->get_type_name()) + " node with name '" + op->get_friendly_name() + "'";
    if (getOriginalInputsNumber() != 1 || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    const auto reorgYolo = std::dynamic_pointer_cast<const ngraph::opset2::ReorgYolo>(op);
    const auto strides = reorgYolo->get_strides();
    if (strides.empty())
        IE_THROW() << errorPrefix << " has empty strides";
    stride = strides[0];
}

void MKLDNNReorgYoloNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32}},
                         {{LayoutType::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void MKLDNNReorgYoloNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

void MKLDNNReorgYoloNode::execute(mkldnn::stream strm) {
    const auto *src_data = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dst_data = reinterpret_cast<float *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());

    const auto &inDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    int IW = (inDims.size() > 3) ? inDims[3] : 1;
    int IH = (inDims.size() > 2) ? inDims[2] : 1;
    int IC = (inDims.size() > 1) ? inDims[1] : 1;
    int B  = (inDims.size() > 0) ? inDims[0] : 1;

    int ic_off = IC / (stride * stride);
    int ih_off = IH * stride;
    int iw_off = IW * stride;
    for (int b = 0; b < B; b++) {
        for (int ic = 0; ic < IC; ic++) {
            for (int ih = 0; ih < IH; ih++) {
                for (int iw = 0; iw < IW; iw++) {
                    int dstIndex = b * IC * IH * IW + ic * IH * IW + ih * IW + iw;

                    int oc = ic % ic_off;
                    int offset = ic / ic_off;

                    int ow = iw * stride + offset % stride;
                    int oh = ih * stride + offset / stride;

                    int srcIndex = b * ic_off * ih_off * iw_off + oc * ih_off * iw_off + oh * iw_off + ow;

                    dst_data[dstIndex] = src_data[srcIndex];
                }
            }
        }
    }
}

bool MKLDNNReorgYoloNode::created() const {
    return getType() == ReorgYolo;
}

REG_MKLDNN_PRIM_FOR(MKLDNNReorgYoloNode, ReorgYolo)
