// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>

#include <ngraph/opsets/opset6.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_experimental_detectron_priorgridgenerator_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNExperimentalDetectronPriorGridGeneratorNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto priorGridGen = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronPriorGridGenerator>(op);
        if (!priorGridGen) {
            errorMessage = "Only opset6 ExperimentalDetectronPriorGridGenerator operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNExperimentalDetectronPriorGridGeneratorNode::MKLDNNExperimentalDetectronPriorGridGeneratorNode
        (const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "ExperimentalDetectronPriorGridGenerator layer with name '" + op->get_friendly_name() + "'";
    const auto priorGridGen = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronPriorGridGenerator>(op);
    if (getOriginalInputsNumber() != 3 || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    if (op->get_input_shape(INPUT_PRIORS).size() != 2 ||
        op->get_input_shape(INPUT_FEATUREMAP).size() != 4 ||
        op->get_input_shape(INPUT_IMAGE).size() != 4)
        IE_THROW() << errorPrefix << " has unsupported input shape";

    const auto &attr = priorGridGen->get_attrs();
    grid_w_ = attr.w;
    grid_h_ = attr.h;
    stride_h_ = attr.stride_y;
    stride_w_ = attr.stride_x;
}

void MKLDNNExperimentalDetectronPriorGridGeneratorNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, Precision::FP32},
                          {TensorDescCreatorTypes::ncsp, Precision::FP32},
                          {TensorDescCreatorTypes::ncsp, Precision::FP32}},
                         {{TensorDescCreatorTypes::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void MKLDNNExperimentalDetectronPriorGridGeneratorNode::execute(mkldnn::stream strm) {
    const int num_priors_ = getParentEdgeAt(INPUT_PRIORS)->getDims()[0];
    assert(getParentEdgeAt(INPUT_PRIORS)->getDims()[1] == 4);

    // Execute
    const int layer_width = grid_w_ ? grid_w_ : getParentEdgeAt(INPUT_FEATUREMAP)->getDims()[3];
    const int layer_height = grid_h_ ? grid_h_ : getParentEdgeAt(INPUT_FEATUREMAP)->getDims()[2];
    const float step_w = stride_w_ ? stride_w_ : static_cast<float>(getParentEdgeAt(INPUT_IMAGE)->getDims()[3]) / layer_width;
    const float step_h = stride_h_ ? stride_h_ : static_cast<float>(getParentEdgeAt(INPUT_IMAGE)->getDims()[2]) / layer_height;

    const auto *bottom_data_0 = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *top_data_0 = reinterpret_cast<float *>(getChildEdgesAtPort(OUTPUT_ROIS)[0]->getMemoryPtr()->GetPtr());

    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            for (int s = 0; s < num_priors_; ++s) {
                top_data_0[0] = bottom_data_0[4 * s + 0] + step_w * (w + 0.5f);
                top_data_0[1] = bottom_data_0[4 * s + 1] + step_h * (h + 0.5f);
                top_data_0[2] = bottom_data_0[4 * s + 2] + step_w * (w + 0.5f);
                top_data_0[3] = bottom_data_0[4 * s + 3] + step_h * (h + 0.5f);
                top_data_0 += 4;
            }
        }
    }
}

bool MKLDNNExperimentalDetectronPriorGridGeneratorNode::created() const {
    return getType() == ExperimentalDetectronPriorGridGenerator;
}

REG_MKLDNN_PRIM_FOR(MKLDNNExperimentalDetectronPriorGridGeneratorNode, ExperimentalDetectronPriorGridGenerator)
