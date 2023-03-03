// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reference.h"
#include <ie_ngraph_utils.hpp>
#include <shape_util.hpp>
#include <dnnl_extension_utils.h>
#include "openvino/runtime/tensor.hpp"
#include "common/blocked_desc_creator.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

Reference::Reference(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context,
                                         const std::string& errorMessage) :
        Node(op, context, NgraphShapeInferFactory(op, FULL_PORT_MASK)), ngraphOp(op), additionalErrorMessage(errorMessage) {
    if (!op->has_evaluate()) {
        IE_THROW(NotImplemented) << "Cannot fallback on ngraph reference implementation (Ngraph::Node::evaluate() is not implemented)";
    }
    setType(Type::Reference);
    setTypeStr("Reference");

    // RandomUniform should generate new sequence each run even if all inputs are constants. So that method Node::IsConstant()
    // doesn't return 'True' for RandomUniform with all constant inputs and the node generates new values for each inference,
    // we set 'NoConst' value for 'ConstantType' in ctor
    if (ov::is_type<ngraph::op::v8::RandomUniform>(ngraphOp)) {
        constant = ConstantType::NoConst;
    }
}

void Reference::getSupportedDescriptors() {}

void Reference::initSupportedPrimitiveDescriptors() {
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

void Reference::createPrimitive() {}

void Reference::execute(dnnl::stream strm) {
    auto inputs = prepareInputs();
    auto outputs = prepareOutputs();
    if (!ngraphOp->evaluate(outputs, inputs)) {
        IE_THROW() << "Evaluation failed on node of type: " << std::string(ngraphOp->get_type_name()) << " name: " << getName();
    }
}

void Reference::executeDynamicImpl(dnnl::stream strm) {
    auto inputs = prepareInputs();
    ov::TensorVector outputs;
    auto result = Node::shapeInfer();
    if (ShapeInferStatus::success == result.status) {
        Node::redefineOutputMemory(result.dims);
        outputs = prepareOutputs();
    } else if (ShapeInferStatus::skip == result.status) {
        outputs.reserve(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); ++i) {
            auto mem_desc = getBaseMemDescAtOutputPort(i);
            if (mem_desc->isDefined()) {
                outputs.emplace_back(ngraphOp->get_output_element_type(i), mem_desc->getShape().getStaticDims());
            } else {
                outputs.emplace_back(ngraphOp->get_output_element_type(i), ov::util::make_dynamic_shape());
            }
        }
    } else {
         IE_THROW(Unexpected) <<
            "Unexpected shape infer result status during the inference of a node with type " <<
            getTypeStr() << " and name " << getName();
    }
    if (!ngraphOp->evaluate(outputs, inputs)) {
        IE_THROW() << "Evaluation failed on node of type: " << std::string(ngraphOp->get_type_name()) << " name: " << getName();
    }
    if (ShapeInferStatus::skip == result.status) {
        std::vector<VectorDims> newOutputDims;
        newOutputDims.reserve(outputs.size());
        for (auto& tensor : outputs) {
            newOutputDims.emplace_back(tensor.get_shape());
        }
        Node::redefineOutputMemory(newOutputDims);
        for (size_t i = 0; i < outputShapes.size(); ++i) {
            auto memory = getChildEdgesAtPort(i)[0]->getMemoryPtr();
            auto& tensor = outputs[i];
            if (memory->GetSize() != tensor.get_byte_size()) {
                IE_THROW(Unexpected) << "Output tensor data size mismatch occurred during the inference of a node with type " <<
                getTypeStr() << " and name " << getName() << " on output port number " << i;
            }
            cpu_memcpy(memory->GetData(), tensor.data(), tensor.get_byte_size());
        }
    }
}

bool Reference::created() const {
    return getType() == Type::Reference;
}

bool Reference::needShapeInfer() const {
    return false;
}

ov::TensorVector Reference::prepareInputs() const {
    ov::TensorVector inputs;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        void *srcDataPtr = getParentEdgesAtPort(i)[0]->getMemory().GetPtr();
        inputs.push_back(ov::Tensor(ngraphOp->get_input_element_type(i),
                                             getParentEdgesAtPort(i)[0]->getMemory().getStaticDims(), srcDataPtr));
    }
    return inputs;
}

ov::TensorVector Reference::prepareOutputs() const {
    ov::TensorVector outputs;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        void *dstDataPtr = getChildEdgesAtPort(i)[0]->getMemory().GetPtr();
        outputs.push_back(ov::Tensor(ngraphOp->get_output_element_type(i),
                                              getChildEdgesAtPort(i)[0]->getMemory().getStaticDims(), dstDataPtr));
    }
    return outputs;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
