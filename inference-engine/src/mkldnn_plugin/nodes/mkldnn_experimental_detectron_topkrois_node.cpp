// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <algorithm>

#include <ngraph/opsets/opset6.hpp>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include "mkldnn_experimental_detectron_topkrois_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNExperimentalDetectronTopKROIsNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto topKROI = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronTopKROIs>(op);
        if (!topKROI) {
            errorMessage = "Only opset6 ExperimentalDetectronTopKROIs operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNExperimentalDetectronTopKROIsNode::MKLDNNExperimentalDetectronTopKROIsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "ExperimentalDetectronTopKROIs layer with name '" + op->get_friendly_name() + "'";
    const auto topKROI = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronTopKROIs>(op);
    if (getOriginalInputsNumber() != 2 || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    if (op->get_input_shape(INPUT_ROIS).size() != 2 || op->get_input_shape(INPUT_PROBS).size() != 1)
        IE_THROW() << errorPrefix << " has nsupported input shape";

    max_rois_num_ = topKROI->get_max_rois();
}

void MKLDNNExperimentalDetectronTopKROIsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{GeneralLayout::ncsp, Precision::FP32},
                          {GeneralLayout::ncsp, Precision::FP32}},
                         {{GeneralLayout::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void MKLDNNExperimentalDetectronTopKROIsNode::execute(mkldnn::stream strm) {
    const int input_rois_num = getParentEdgeAt(INPUT_ROIS)->getShape().getStaticDims()[0];
    const int top_rois_num = (std::min)(max_rois_num_, input_rois_num);

    auto *input_rois = reinterpret_cast<const float *>(getParentEdgeAt(INPUT_ROIS)->getMemoryPtr()->GetPtr());
    auto *input_probs = reinterpret_cast<const float *>(getParentEdgeAt(INPUT_PROBS)->getMemoryPtr()->GetPtr());
    auto *output_rois = reinterpret_cast<float *>(getChildEdgesAtPort(OUTPUT_ROIS)[0]->getMemoryPtr()->GetPtr());

    std::vector<size_t> idx(input_rois_num);
    iota(idx.begin(), idx.end(), 0);
    // FIXME. partial_sort is enough here.
    sort(idx.begin(), idx.end(), [&input_probs](size_t i1, size_t i2) {return input_probs[i1] > input_probs[i2];});

    for (int i = 0; i < top_rois_num; ++i) {
        cpu_memcpy(output_rois + 4 * i, input_rois + 4 * idx[i], 4 * sizeof(float));
    }
}

bool MKLDNNExperimentalDetectronTopKROIsNode::created() const {
    return getType() == ExperimentalDetectronTopKROIs;
}

REG_MKLDNN_PRIM_FOR(MKLDNNExperimentalDetectronTopKROIsNode, ExperimentalDetectronTopKROIs)
