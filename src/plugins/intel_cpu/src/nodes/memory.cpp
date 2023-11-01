// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include "memory.hpp"
#include "common/cpu_convert.h"
#include "common/cpu_memcpy.h"
#include "utils/general_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/ngraph_utils.hpp"

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

std::mutex MemoryNodeVirtualEdge::holderMutex;

MemoryNode::MemoryNode(const std::shared_ptr<ov::Node>& op) {
    if (auto assignOp = std::dynamic_pointer_cast<ngraph::op::AssignBase>(op)) {
        _id = assignOp->get_variable_id();
    } else if (auto readValueOp = std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(op)) {
        _id = readValueOp->get_variable_id();
    }
}

bool MemoryOutput::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }

        if (!one_of(op->get_type_info(),
                ngraph::op::v3::Assign::get_type_info_static(),
                ngraph::op::v6::Assign::get_type_info_static())) {
            errorMessage = "Node is not an instance of Assign from the operation set v3 or v6.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MemoryOutput::MemoryOutput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) , MemoryNode(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (created()) {
        holder = MemoryNodeVirtualEdge::registerOutput(this);
    }
}

MemoryOutput::~MemoryOutput() {
    MemoryNodeVirtualEdge::remove(this, holder);
}

void MemoryOutput::getSupportedDescriptors() {}

void MemoryOutput::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    NodeConfig config;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace(-1);
    config.inConfs[0].constant(false);
    config.inConfs[0].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(precision, getInputShapeAtPort(0)));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MemoryOutput::execute(dnnl::stream strm)  {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    auto inputMemoryNode = dynamic_cast<MemoryInput*>(inputNode);
    IE_ASSERT(inputMemoryNode != nullptr);
    inputMemoryNode->storeState(srcMemory);
}

bool MemoryInput::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }

        if (!one_of(op->get_type_info(),
                ngraph::op::v3::ReadValue::get_type_info_static(),
                ngraph::op::v6::ReadValue::get_type_info_static())) {
            errorMessage = "Node is not an instance of ReadValue from the operation set v3 or v6.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MemoryInput::MemoryInput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr ctx)
        : Input(op, ctx), MemoryNode(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (created()) {
        holder = MemoryNodeVirtualEdge::registerInput(this);
    }
}

void MemoryInput::createPrimitive() {
    Input::createPrimitive();

    dataStore = std::make_shared<Memory>(getEngine(), getChildEdgeAt(0)->getMemory().getDesc());

    // default memory state is zero filled
    if (dataStore->getDesc().hasDefinedMaxSize())
        dataStore->nullify();
}

/**
 * Copy data from one tensor into other.
 * As is. Assume that data is dense tensor with same layout.
 * @param dst destination memory object
 * @param src source memory object
 */
inline
static void simple_copy(const IMemory& dst, const IMemory& src) {
    auto srcPtr = static_cast<uint8_t*>(src.getData());
    auto dstPtr = static_cast<uint8_t*>(dst.getData());
    if (src.getDataType() == dst.getDataType()) {
        auto srcSizeInByte = src.getSize();
        auto dstSizeInByte = dst.getSize();

        IE_ASSERT(srcSizeInByte == dstSizeInByte) << "MemoryNode objects are not compatible. Has different sizes.";

        cpu_memcpy(dstPtr, srcPtr, srcSizeInByte);
    } else {
        cpu_convert(srcPtr, dstPtr, src.getDesc().getPrecision(),
            dst.getDesc().getPrecision(), src.getDesc().getShape().getElementsCount());
    }
}

MemoryInput::~MemoryInput() {
    MemoryNodeVirtualEdge::remove(this, holder);
}

MemoryPtr MemoryInput::getStore() {
    return dataStore;
}

void MemoryInput::storeState(const IMemory &new_state) {
    // TODO: Should be next one call:
    //           dataStore.load(new_state, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(*dataStore, new_state);
}

void MemoryInput::execute(dnnl::stream strm) {
    // TODO: Should be simple call of:
    //           dst_mem.load(dataStore, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(getChildEdgeAt(0)->getMemory(), *dataStore);
}

MemoryNodeVirtualEdge::Holder* MemoryNodeVirtualEdge::registerInput(MemoryInput * node) {
    std::lock_guard<std::mutex> lock{MemoryNodeVirtualEdge::holderMutex};
    // in case of output already registered
    auto& holder = MemoryNodeVirtualEdge::getExisted();
    auto sibling = MemoryNodeVirtualEdge::getByName(holder, node->getId());
    if (sibling != nullptr) {
        auto outputNode = dynamic_cast<MemoryOutput*>(sibling);
        IE_ASSERT(outputNode != nullptr);
        outputNode->setInputNode(node);
    } else {
        holder[node->getId()] = node;
    }
    return &holder;
}

MemoryNodeVirtualEdge::Holder* MemoryNodeVirtualEdge::registerOutput(MemoryOutput * node) {
    std::lock_guard<std::mutex> lock{MemoryNodeVirtualEdge::holderMutex};
    // in case of output layer
    auto& holder = MemoryNodeVirtualEdge::getExisted();
    auto sibling = MemoryNodeVirtualEdge::getByName(holder, node->getId());
    if (sibling != nullptr) {
        auto inputNode = dynamic_cast<MemoryInput*>(sibling);
        IE_ASSERT(inputNode != nullptr);
        node->setInputNode(inputNode);
    } else {
        holder[node->getId()] = node;
    }
    return &holder;
}

void MemoryNodeVirtualEdge::remove(MemoryNode * node, Holder* holder) {
    std::lock_guard<std::mutex> lock{MemoryNodeVirtualEdge::holderMutex};
    if (nullptr != holder) {
        InferenceEngine::details::erase_if(*holder, [&](const Holder::value_type & it){
            return it.second == node;
        });
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
