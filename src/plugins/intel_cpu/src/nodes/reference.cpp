// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reference.h"

#include "common/cpu_memcpy.h"
#include <ie_ngraph_utils.hpp>
#include "openvino/core/shape_util.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

Reference::Reference(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context,
                                         const std::string& errorMessage) :
        Node(op, context, NgraphShapeInferFactory(op, FULL_PORT_MASK)), ovCoreNode(op), additionalErrorMessage(errorMessage) {
    if (!op->has_evaluate()) {
        IE_THROW(NotImplemented) << "Cannot fallback on ngraph reference implementation (Ngraph::Node::evaluate() is not implemented)";
    }

    setType(Type::Reference);
    setTypeStr("Reference");
}

void Reference::getSupportedDescriptors() {}

void Reference::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inputConfigurators;
    inputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); i++) {
        inputConfigurators.emplace_back(LayoutType::ncsp, convertPrecision(ovCoreNode->get_input_element_type(i)), inputShapes[i]);
    }

    std::vector<PortConfigurator> outputConfigurators;
    outputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); i++) {
        outputConfigurators.emplace_back(LayoutType::ncsp, convertPrecision(ovCoreNode->get_output_element_type(i)), outputShapes[i]);
    }

    addSupportedPrimDesc(inputConfigurators, outputConfigurators, impl_desc_type::ref);
}

void Reference::createPrimitive() {}

void Reference::execute(dnnl::stream strm) {
    auto inputs = prepareInputs();
    auto outputs = prepareOutputs();
    if (!ovCoreNode->evaluate(outputs, inputs)) {
        THROW_CPU_NODE_ERR("evaluation failed for core operation: ", std::string(ovCoreNode->get_type_name()));
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
                outputs.emplace_back(ovCoreNode->get_output_element_type(i), mem_desc->getShape().getStaticDims());
            } else {
                outputs.emplace_back(ovCoreNode->get_output_element_type(i), ov::util::make_dynamic_shape());
            }
        }
    } else {
         THROW_CPU_NODE_ERR("got unexpected shape infer result status during the inference.");
    }
    if (!ovCoreNode->evaluate(outputs, inputs)) {
        THROW_CPU_NODE_ERR("evaluation failed for core operation: ", std::string(ovCoreNode->get_type_name()));
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
            if (memory->getSize() != tensor.get_byte_size()) {
                THROW_CPU_NODE_ERR("output tensor data size mismatch occurred during the inference on output port number ", i);
            }
            cpu_memcpy(memory->getData(), tensor.data(), tensor.get_byte_size());
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
        void *srcDataPtr = getParentEdgesAtPort(i)[0]->getMemory().getData();
        ov::Shape shape = ovCoreNode->get_input_partial_shape(i).rank().get_length() == 0 ?
                ov::Shape{} : getParentEdgesAtPort(i)[0]->getMemory().getStaticDims();
        inputs.push_back(ov::Tensor(ovCoreNode->get_input_element_type(i), shape, srcDataPtr));
    }
    return inputs;
}

ov::TensorVector Reference::prepareOutputs() const {
    ov::TensorVector outputs;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        void *dstDataPtr = getChildEdgesAtPort(i)[0]->getMemory().getData();
        ov::Shape shape = ovCoreNode->get_output_partial_shape(i).rank().get_length() == 0 ?
                ov::Shape{} : getChildEdgesAtPort(i)[0]->getMemory().getStaticDims();
        if (dstDataPtr != nullptr) {
            outputs.push_back(ov::Tensor(ovCoreNode->get_output_element_type(i), shape, dstDataPtr));
        } else {
            outputs.push_back(ov::Tensor(ovCoreNode->get_output_element_type(i), shape));
        }
    }
    return outputs;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
