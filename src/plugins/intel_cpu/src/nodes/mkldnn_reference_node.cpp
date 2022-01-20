// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reference_node.h"
#include <ie_ngraph_utils.hpp>
#include <mkldnn_extension_utils.h>
#include "openvino/runtime/tensor.hpp"
#include "common/blocked_desc_creator.h"
#include <ngraph/opsets/opset1.hpp>

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

    // RandomUniform should generate new sequence each run even if all inputs are constants. So that method MKLDNNNode::IsConstant()
    // doesn't return 'True' for RandomUniform with all constant inputs and the node generates new values for each inference,
    // we set 'NoConst' value for 'ConstantType' in ctor
    if (ov::is_type<ngraph::op::v8::RandomUniform>(ngraphOp)) {
        constant = ConstantType::NoConst;
    }
}

void MKLDNNReferenceNode::getSupportedDescriptors() {}

void MKLDNNReferenceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inputConfigurators;
    inputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); i++) {
        inputConfigurators.emplace_back(LayoutType::ncsp, convertPrecision(ngraphOp->get_input_element_type(i)), inputShapes[i]);
    }

    std::vector<PortConfigurator> outputConfigurators;
    outputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); i++) {
        outputConfigurators.emplace_back(LayoutType::ncsp, convertPrecision(ngraphOp->get_output_element_type(i)), outputShapes[i]);
    }

    addSupportedPrimDesc(inputConfigurators, outputConfigurators, impl_desc_type::ref);
}

void MKLDNNReferenceNode::createPrimitive() {}

void MKLDNNReferenceNode::execute(mkldnn::stream strm) {
    ov::runtime::TensorVector inputs;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        void *srcDataPtr = getParentEdgesAtPort(i)[0]->getMemory().GetPtr();
        inputs.push_back(ov::runtime::Tensor(ngraphOp->get_input_element_type(i),
                                             getParentEdgesAtPort(i)[0]->getMemory().getStaticDims(), srcDataPtr));
    }

    ov::runtime::TensorVector outputs;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        void *dstDataPtr = getChildEdgesAtPort(i)[0]->getMemory().GetPtr();
        outputs.push_back(ov::runtime::Tensor(ngraphOp->get_output_element_type(i),
                                              getChildEdgesAtPort(i)[0]->getMemory().getStaticDims(), dstDataPtr));
    }

    if (!ngraphOp->evaluate(outputs, inputs)) {
        IE_THROW() << "Evaluation failed on node of type: " << std::string(ngraphOp->get_type_name()) << " name: " << getName();
    }
}

std::vector<VectorDims> MKLDNNReferenceNode::shapeInfer() const {
    return MKLDNNNode::shapeInferGeneric(0xFFFFFFFF);
}

void MKLDNNReferenceNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNReferenceNode::created() const {
    return getType() == Reference;
}

bool MKLDNNReferenceNode::needShapeInfer() const {
    return true;
}
