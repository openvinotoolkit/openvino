// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include <ngraph/opsets/opset1.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_grn_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNGRNNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto grn = std::dynamic_pointer_cast<const ngraph::opset1::GRN>(op);
        if (!grn) {
            errorMessage = "Only opset1 GRN operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNGRNNode::MKLDNNGRNNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "GRN layer with name '" + op->get_friendly_name() + "'";
    const auto grn = std::dynamic_pointer_cast<const ngraph::opset1::GRN>(op);

    if (getOriginalInputsNumber() != 1 || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    bias = grn->get_bias();
}

void MKLDNNGRNNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{GeneralLayout::ncsp, Precision::FP32, false, 0}},
                         {{GeneralLayout::ncsp, Precision::FP32, false, 0}},
                         impl_desc_type::ref_any);
}

void MKLDNNGRNNode::execute(mkldnn::stream strm) {
    const float* src_data = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    float* dst_data = reinterpret_cast<float *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());

    SizeVector dims = getParentEdgeAt(0)->getShape().getStaticDims();

    int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
    int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
    int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
    int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

    parallel_for3d(N, H, W, [&](int b, int h, int w) {
        double variance = 0;
        for (int c = 0; c < C; c++) {
            variance += std::pow(src_data[b*C*H*W + c*H*W + h*W + w], 2);
        }
        variance = std::pow(variance + bias, 0.5f);
        for (int c = 0; c < C; c++) {
            dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] / static_cast<float>(variance);
        }
    });
}

bool MKLDNNGRNNode::created() const {
    return getType() == GRN;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGRNNode, GRN)
