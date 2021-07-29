// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reference_node.h"
#include <ie_ngraph_utils.hpp>
#include <mkldnn_extension_utils.h>
#include <ngraph/runtime/host_tensor.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

MKLDNNReferenceNode::MKLDNNReferenceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache,
                                         const std::string& errorMessage) :
        MKLDNNNode(op, eng, cache), ngraphOp(op), additionalErrorMessage(errorMessage) {
    if (!op->has_evaluate()) {
        IE_THROW(NotImplemented) << "Cannot fallback on ngraph reference implementation (Ngraph::Node::evaluate() is not implemented)";
    }
    setType(Reference);
    setTypeStr("Reference");
}

void MKLDNNReferenceNode::getSupportedDescriptors() {}

void MKLDNNReferenceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::LayerConfig config;
    for (size_t i = 0; i < inDims.size(); i++) {
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        dataConfig.desc = MKLDNNMemoryDesc(inDims[i],
                MKLDNNExtensionUtils::IEPrecisionToDataType(convertPrecision(ngraphOp->get_input_element_type(i))),
                MKLDNNMemory::GetPlainFormat(inDims[i]));

        config.inConfs.push_back(dataConfig);
    }

    for (size_t i = 0; i < outDims.size(); i++) {
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        dataConfig.desc = MKLDNNMemoryDesc(outDims[i],
                MKLDNNExtensionUtils::IEPrecisionToDataType(convertPrecision(ngraphOp->get_output_element_type(i))),
                MKLDNNMemory::GetPlainFormat(outDims[i]));

        config.outConfs.push_back(dataConfig);
    }

    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref, memory::format_tag::undef});
}

void MKLDNNReferenceNode::createPrimitive() {}

void MKLDNNReferenceNode::execute(mkldnn::stream strm) {
    ngraph::HostTensorVector inputs;
    for (size_t i = 0; i < inDims.size(); i++) {
        void *srcDataPtr = getParentEdgesAtPort(i)[0]->getMemory().GetPtr();
        inputs.push_back(std::make_shared<ngraph::HostTensor>(ngraphOp->get_input_element_type(i), ngraphOp->get_input_shape(i), srcDataPtr));
    }

    ngraph::HostTensorVector outputs;
    for (size_t i = 0; i < outDims.size(); i++) {
        void *dstDataPtr = getChildEdgesAtPort(i)[0]->getMemory().GetPtr();
        outputs.push_back(std::make_shared<ngraph::HostTensor>(ngraphOp->get_output_element_type(i), ngraphOp->get_output_shape(i), dstDataPtr));
    }

    if (!ngraphOp->evaluate(outputs, inputs)) {
        IE_THROW() << "Evaluation failed on node of type: " << std::string(ngraphOp->get_type_name()) << " name: " << getName();
    }
}

bool MKLDNNReferenceNode::created() const {
    return getType() == Reference;
}
