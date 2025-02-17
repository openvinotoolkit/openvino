// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reference.h"

#include <utility>

#include "common/cpu_memcpy.h"
#include "shape_inference/shape_inference.hpp"

namespace ov::intel_cpu::node {

Reference::Reference(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context, std::string errorMessage)
    : Node(op, context, NgraphShapeInferFactory(op)),
      ovCoreNode(op),
      additionalErrorMessage(std::move(errorMessage)) {
    if (!op->has_evaluate()) {
        OPENVINO_THROW_NOT_IMPLEMENTED(
            "Cannot fallback on ngraph reference implementation. Ngraph::Node::evaluate() is not implemented for op: ",
            *op);
    }

    setType(Type::Reference);
    setTypeStr("Reference");
}

void Reference::getSupportedDescriptors() {}

void Reference::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    std::vector<PortConfigurator> inputConfigurators;
    inputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); i++) {
        inputConfigurators.emplace_back(LayoutType::ncsp, ovCoreNode->get_input_element_type(i), inputShapes[i]);
    }

    std::vector<PortConfigurator> outputConfigurators;
    outputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); i++) {
        outputConfigurators.emplace_back(LayoutType::ncsp, ovCoreNode->get_output_element_type(i), outputShapes[i]);
    }

    addSupportedPrimDesc(inputConfigurators, outputConfigurators, impl_desc_type::ref);
}

void Reference::createPrimitive() {
    hasOutputShapeDataDependency = isDynamicNode() && outputShapeDataDependency();
}

void Reference::execute(const dnnl::stream& strm) {
    auto inputs = prepareInputs();
    auto outputs = prepareOutputs();
    if (!ovCoreNode->evaluate(outputs, inputs)) {
        THROW_CPU_NODE_ERR("evaluation failed for core operation: ", std::string(ovCoreNode->get_type_name()));
    }
}

void Reference::executeDynamicImpl(const dnnl::stream& strm) {
    if (!hasOutputShapeDataDependency) {
        // if there is no data dependency for the output shape, we can execute the operation as is, similar to the
        // static case, since the shapes are already calculated
        execute(strm);
        return;
    }

    // if there is data dependency, we need to perform shape inference first
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
                outputs.emplace_back(ovCoreNode->get_output_element_type(i), ov::Shape{0});
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
            auto memory = getDstMemoryAtPort(i);
            auto& tensor = outputs[i];
            if (memory->getSize() != tensor.get_byte_size()) {
                THROW_CPU_NODE_ERR(
                    "output tensor data size mismatch occurred during the inference on output port number ",
                    i);
            }
            if (tensor.get_element_type() == element::string) {
                auto srcPtr = tensor.data<StringMemory::OvString>();
                auto dstPtr = memory->getDataAs<StringMemory::OvString>();
                std::copy(srcPtr, srcPtr + tensor.get_size(), dstPtr);
            } else {
                cpu_memcpy(memory->getData(), tensor.data(), tensor.get_byte_size());
            }
        }
    }
}

bool Reference::created() const {
    return getType() == Type::Reference;
}

bool Reference::needShapeInfer() const {
    // If there is data dependency for the output shape, let's assume the node has internal dynamism (in general case),
    // so we postpone the shape inference until the actual execution
    return !hasOutputShapeDataDependency && Node::needShapeInfer();
}

ov::TensorVector Reference::prepareInputs() const {
    ov::TensorVector inputs;
    for (size_t i = 0lu; i < inputShapes.size(); i++) {
        void* srcDataPtr = getSrcDataAtPort(i);
        ov::Shape shape = ovCoreNode->get_input_partial_shape(i).rank().get_length() == 0
                              ? ov::Shape{}
                              : getParentEdgeAt(i)->getMemory().getStaticDims();

        if (std::any_of(shape.begin(), shape.end(), [](const size_t dim) {
                return dim == 0lu;
            })) {
            inputs.emplace_back(ovCoreNode->get_input_element_type(i), shape);
        } else {
            CPU_NODE_ASSERT(srcDataPtr, "has empty input data on port ", i);
            inputs.emplace_back(ovCoreNode->get_input_element_type(i), shape, srcDataPtr);
        }
    }
    return inputs;
}

ov::TensorVector Reference::prepareOutputs() const {
    ov::TensorVector outputs;
    for (size_t i = 0lu; i < outputShapes.size(); i++) {
        void* dstDataPtr = getDstDataAtPort(i);
        ov::Shape shape = ovCoreNode->get_output_partial_shape(i).rank().get_length() == 0
                              ? ov::Shape{}
                              : getChildEdgeAt(i)->getMemory().getStaticDims();

        if (std::any_of(shape.begin(), shape.end(), [](const size_t dim) {
                return dim == 0lu;
            })) {
            outputs.emplace_back(ovCoreNode->get_output_element_type(i), shape);
        } else {
            CPU_NODE_ASSERT(dstDataPtr, "has empty output data on port ", i);
            outputs.emplace_back(ovCoreNode->get_output_element_type(i), shape, dstDataPtr);
        }
    }
    return outputs;
}

}  // namespace ov::intel_cpu::node
