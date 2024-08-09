// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input.h"

#include "cpu_memory.h"
#include "openvino/op/constant.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

static MemoryPtr createMemoryForConstantOp(const std::shared_ptr<ov::op::v0::Constant>& constOp, dnnl::engine engine) {
    Shape shape(constOp->get_shape().empty() ? ov::Shape(1, 1) : constOp->get_shape());
    CpuBlockedMemoryDesc memDesc(constOp->get_element_type(), shape);
    return std::make_shared<Memory>(engine, memDesc, constOp->get_data_ptr());
}

Input::Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    if (!one_of(op->get_type_info(),
                op::v0::Parameter::get_type_info_static(),
                op::v0::Constant::get_type_info_static(),
                op::v0::Result::get_type_info_static(),
                op::v3::ReadValue::get_type_info_static(),
                op::v6::ReadValue::get_type_info_static()))
        OPENVINO_THROW_NOT_IMPLEMENTED("CPU Input node doesn't support ngraph operation ",
                                       op->get_type_name(),
                                       " with name ",
                                       op->get_friendly_name());
    // @todo is it required to hold a pointer to the original Constant to preserve a memory?
    constOp = ov::as_type_ptr<op::v0::Constant>(op);

    if (constOp) {
        constant = ConstantType::Const;
        memoryPtr = createMemoryForConstantOp(constOp, getEngine());
    } else {
        constant = ConstantType::StrictNoConst;
    }
}

static std::vector<Shape> createInputShapes(const Shape& shape,
                                            const Type type) {
    if (type == Type::Output)
        return {shape};
    return {};
}

static std::vector<Shape> createOutputShapes(const Shape& shape,
                                             const Type type) {
    if (type == Type::Input)
        return {shape};
    return {};
}

static std::vector<ov::element::Type> createInputPrecisions(const ov::element::Type& prc,
                                                         const Type type) {
    if (type == Type::Output)
        return {prc};
    return {};
}

static std::vector<ov::element::Type> createOutputPrecisions(const ov::element::Type& prc,
                                                          const Type type) {
    if (type == Type::Input)
        return {prc};
    return {};
}

Input::Input(const Shape& shape,
             const ov::element::Type& prc,
             const std::string& name,
             const std::string& type,
             const GraphContext::CPtr context)
    : Node(type,
           createInputShapes(shape, TypeFromName(type)),
           createOutputShapes(shape, TypeFromName(type)),
           createInputPrecisions(prc, TypeFromName(type)),
           createOutputPrecisions(prc, TypeFromName(type)),
           name,
           context) {
    constant = ConstantType::NoConst;
    isDynamic = shape.isDynamic();
    if (isDynamic) {
        shapeInference = PassThroughShapeInferFactory().makeShapeInfer();
    }
}

Input::Input(MemoryDescPtr memDesc, const std::string& name, const std::string& type, const GraphContext::CPtr context)
    : Input(memDesc->getShape(), memDesc->getPrecision(), name, type, context) {
    extMemDesc = memDesc;
}

MemoryCPtr Input::getMemoryPtr() const {
    return memoryPtr;
}

void Input::getSupportedDescriptors() {
    if (getType() == Type::Input) {
        if (!getParentEdges().empty())
            THROW_CPU_NODE_ERR("has incorrect number of input edges.");
        if (getChildEdges().empty())
            THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    } else if (getType() == Type::Output) {
        if (getParentEdges().size() != 1)
            THROW_CPU_NODE_ERR("has incorrect number of input edges.");
        if (!getChildEdges().empty())
            THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void Input::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    if (extMemDesc) {
        initSupportedPdFromMemDesc();
    } else {
        initSupportedPdDefault();
    }
}

void Input::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto dstMemPtr = getDstMemoryAtPort(i);
        if (!dstMemPtr || !dstMemPtr->isAllocated())
            THROW_CPU_NODE_ERR("has unallocated memory object at port ", i,
                              " to node ", getChildEdgeAt(i)->getChild()->getName(), ".");
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto srcMemPtr = getSrcMemoryAtPort(i);
        if (!srcMemPtr || !srcMemPtr->isAllocated())
            THROW_CPU_NODE_ERR("has unallocated memory object at port ", i,
                              " from node ", getParentEdgeAt(i)->getParent()->getName(), ".");
    }

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_CPU_NODE_ERR("doesn't have selected primitive descriptor.");
}

bool Input::created() const {
    return getType() == Type::Input || getType() == Type::Output;
}

void Input::initSupportedPdDefault() {
    std::vector<PortConfigurator> inPortConfs;
    std::vector<PortConfigurator> outPortConfs;

    if (getType() == Type::Input || getType() == Type::MemoryInput) {
        auto precision = getOriginalOutputPrecisionAtPort(0);

        outPortConfs.push_back({LayoutType::ncsp, precision});
        if (!getParentEdges().empty()) {
            inPortConfs.push_back({LayoutType::ncsp, precision, true});
        }
    } else if (getType() == Type::Output) {
        auto precision = getOriginalInputPrecisionAtPort(0);

        inPortConfs.push_back({LayoutType::ncsp, precision});
    }

    addSupportedPrimDesc(inPortConfs,
                         outPortConfs,
                         impl_desc_type::unknown);
}

void Input::initSupportedPdFromMemDesc() {
    NodeConfig config;
    PortConfig portConfig;
    portConfig.inPlace(-1);
    portConfig.constant(false);
    portConfig.setMemDesc(extMemDesc);
    if (getType() == Type::Input || getType() == Type::MemoryInput) {
        config.outConfs.push_back(portConfig);
    } else if (getType() == Type::Output) {
        config.inConfs.push_back(portConfig);
    }
    supportedPrimitiveDescriptors.emplace_back(std::move(config), impl_desc_type::unknown);
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
