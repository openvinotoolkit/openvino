// Copyright (C) 2021 Intel Corporation
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

// TODO [DS]: rewrite after new shape infer will be added
std::vector<VectorDims> MKLDNNReferenceNode::shapeInfer() const {
    ngraph::OutputVector inputsForShapeInfer;
    for (size_t i = 0; i < opToShapeInfer->get_input_size(); i++) {
        const auto &mem = getParentEdgesAtPort(i)[0]->getMemory();
        const auto dims = opToShapeInfer->get_input_partial_shape(i).rank().get_length() == 0 ? VectorDims{} : mem.getStaticDims();
        inputsForShapeInfer.push_back(std::make_shared<ngraph::opset1::Constant>(InferenceEngine::details::convertPrecision(mem.getDesc().getPrecision()),
                                                                                 dims,
                                                                                 mem.GetPtr()));
    }

    const auto localShapeInferOp = opToShapeInfer->clone_with_new_inputs(inputsForShapeInfer);
    localShapeInferOp->validate_and_infer_types();

    std::vector<VectorDims> newOutputShapes(outputShapes.size());
    for (size_t i = 0; i < newOutputShapes.size(); i++) {
        const auto &partShape = localShapeInferOp->get_output_partial_shape(i);
        if (partShape.is_dynamic()) {
            std::ostringstream errorMessage;
            errorMessage << "Can't compute static output shape on " << i << " port for node with name: " << getName();
            errorMessage << ". Input shapes = ( ";
            for (size_t in = 0; in < opToShapeInfer->get_input_size(); in++) {
                errorMessage << in << " port = " << opToShapeInfer->get_input_partial_shape(in) << ", ";
            }
            errorMessage << "). Output shapes = ( ";
            for (size_t out = 0; out < opToShapeInfer->get_output_size(); out++) {
                errorMessage << out << " port = " << opToShapeInfer->get_output_partial_shape(out) << ", ";
            }
            errorMessage << ")";
            IE_THROW(NotImplemented) << errorMessage.str();
        }
        newOutputShapes[i] = partShape.get_shape();
    }
    return newOutputShapes;
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
