// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_lrn_node.h"
#include <string>
#include <mkldnn_extension_utils.h>
#include <ngraph/opsets/opset1.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNLrnNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto lrn = std::dynamic_pointer_cast<const ngraph::opset1::LRN>(op);
        if (!lrn) {
            errorMessage = "Only opset1 LRN operation is supported";
            return false;
        }

        const auto dataDims = lrn->get_input_shape(0);
        if (dataDims.size() < 2 || dataDims.size() > 5) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(dataDims.size());
            return false;
        }
        const auto axesNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(lrn->get_input_node_shared_ptr(1));
        if (!axesNode) {
            errorMessage = "Only Constant operation on 'axis' input is supported";
            return false;
        }

        const auto axes = axesNode->cast_vector<int64_t>();
        const auto dataRank = dataDims.size();
        if (axes.size() == 1 && axes[0] == 1) {
            return true;
        } else {
            std::vector<bool> norm(dataRank, false);
            for (auto &axis : axes) {
                if (axis < 0 || axis >= dataRank) {
                    errorMessage = "Has incorrect reduction axis: " + std::to_string(axis);
                    return false;
                }
                norm[axis] = true;
            }

            for (size_t i = 2; i < norm.size(); ++i) {
                if (!norm[i]) {
                    errorMessage = "Supports only across channels or across spatial reduction";
                    return false;
                }
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNLrnNode::MKLDNNLrnNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "LRN node with name '" + getName() + "'";

        const auto lrn = std::dynamic_pointer_cast<const ngraph::opset1::LRN>(op);
        const auto axes = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(lrn->get_input_node_shared_ptr(1))->cast_vector<int64_t>();
        isAcrossMaps = (axes.size() == 1 && axes[0] == 1);
        alpha = static_cast<float>(lrn->get_alpha());
        beta = static_cast<float>(lrn->get_beta());
        k = static_cast<float>(lrn->get_bias());
        size = lrn->get_nsize();
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNLrnNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has incorrect number of output edges";

    InferenceEngine::Precision precision = getOriginalOutputPrecisionAtPort(0);
    if (precision != InferenceEngine::Precision::FP32 && precision != InferenceEngine::Precision::BF16)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto parentDims = getParentEdgeAt(0)->getDims();

    for (auto format : getAvailableFormatsForDims(parentDims)) {
        MKLDNNMemoryDesc in_candidate(parentDims, inputDataType, format);
        createDescriptor({in_candidate}, {});
    }
}

MKLDNNMemoryDesc MKLDNNLrnNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    if (idx > 0) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(getOriginalInputPrecisions()[idx],
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            TensorDesc::getLayoutByDims(getParentEdgeAt(idx)->getDims().ToSizeVector())));
    } else {
        return MKLDNNNode::getSrcMemDesc(primitive_desc_it, idx);
    }
}

void MKLDNNLrnNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<mkldnn::lrn_forward::primitive_desc, mkldnn::lrn_forward::desc>();

    prim.reset(new mkldnn::lrn_forward(prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};
}

bool MKLDNNLrnNode::created() const {
    return getType() == Lrn;
}

void MKLDNNLrnNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                     const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    mkldnn::algorithm alg = isAcrossMaps ? mkldnn::algorithm::lrn_across_channels : mkldnn::algorithm::lrn_within_channel;
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNDescriptor desc(std::shared_ptr<mkldnn::lrn_forward::desc>(
            new mkldnn::lrn_forward::desc(mkldnn::prop_kind::forward_scoring, alg, in_candidate, size, alpha, beta, k)));
    descs.push_back(desc);
}

REG_MKLDNN_PRIM_FOR(MKLDNNLrnNode, Lrn);
