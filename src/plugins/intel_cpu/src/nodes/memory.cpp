// Copyright (C) 2018-2022 Intel Corporation
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

MemoryNode::MemoryNode(const std::string & id) : _id(id) {}

MemoryNode::MemoryNode(const std::shared_ptr<ngraph::Node>& op) {
    if (auto assignOp = std::dynamic_pointer_cast<ngraph::op::AssignBase>(op)) {
        _id = assignOp->get_variable_id();
    } else if (auto readValueOp = std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(op)) {
        _id = readValueOp->get_variable_id();
    }
}

const std::string & MemoryNode::getId() const {
    return _id;
}

bool MemoryOutput::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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

MemoryOutput::MemoryOutput(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) , MemoryNode(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

MemoryOutput::~MemoryOutput() {
    unregisterThis();
}

void MemoryOutput::getSupportedDescriptors() {}

void MemoryOutput::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace(-1);
    config.inConfs[0].constant(false);
    config.inConfs[0].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(precision, getInputShapeAtPort(0)));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MemoryOutput::execute(dnnl::stream strm)  {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    if (auto inputMemoryNode = _inputNode.lock()) {
        inputMemoryNode->storeState(srcMemory);
    } else {
        IE_THROW(GeneralError) << "Input memory node is null";
    }
}

bool MemoryOutput::created() const {
    return getType() == Type::MemoryOutput;
}

void MemoryOutput::setInputNode(const std::weak_ptr<MemoryInput> & node) {
    _inputNode = node;
}

void MemoryOutput::registerThis(const NodesUnorderedMapPtr & memoryNodes) {
    _memoryNodes = memoryNodes;

    if (auto memoryNodes = _memoryNodes.lock()) {
        auto it = memoryNodes->find(getId());
        if (it != memoryNodes->end()) {
            auto inputNode = std::dynamic_pointer_cast<MemoryInput>(it->second);
            IE_ASSERT(inputNode != nullptr);
            setInputNode(inputNode);
        } else {
            auto outputNode = std::static_pointer_cast<MemoryOutput>(shared_from_this());
            memoryNodes->emplace(getId(), std::static_pointer_cast<Node>(outputNode));
        }
    }
}

void MemoryOutput::unregisterThis() {
    if (auto memoryNodes = _memoryNodes.lock()) {
        auto it = memoryNodes->find(getId());
        if (it->second.get() == this) {
            memoryNodes->erase(it);
        }
    }
}

bool MemoryInput::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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

MemoryInput::MemoryInput(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Input(op, eng, cache), MemoryNode(op), dataStore(new Memory{eng}) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MemoryInput::createPrimitive() {
    Input::createPrimitive();

    dataStore->Create(getChildEdgeAt(0)->getMemory().getDesc());

    // default memory state is zero filled
    if (dataStore->getDesc().hasDefinedMaxSize())
        dataStore->FillZero();
}

/**
 * Copy data from one tensor into other.
 * As is. Assume that data is dense tensor with same layout.
 * @param dst destination memory object
 * @param src source memory object
 */
inline
static void simple_copy(const Memory& dst, const Memory& src) {
    auto srcPtr = static_cast<uint8_t*>(src.GetPtr());
    auto dstPtr = static_cast<uint8_t*>(dst.GetPtr());
    if (src.GetDataType() == dst.GetDataType()) {
        auto srcSizeInByte = src.GetSize();
        auto dstSizeInByte = dst.GetSize();

        IE_ASSERT(srcSizeInByte == dstSizeInByte) << "MemoryNode objects are not compatible. Has different sizes.";

        cpu_memcpy(dstPtr, srcPtr, srcSizeInByte);
    } else {
        cpu_convert(srcPtr, dstPtr, src.getDesc().getPrecision(),
            dst.getDesc().getPrecision(), src.getDesc().getShape().getElementsCount());
    }
}

MemoryInput::~MemoryInput() {
    unregisterThis();
}

MemoryPtr MemoryInput::getStore() {
    return dataStore;
}

void MemoryInput::storeState(const Memory &new_state) {
    // TODO: Should be next one call:
    //           dataStore.SetData(new_state, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(*dataStore, new_state);
}

void MemoryInput::execute(dnnl::stream strm) {
    // TODO: Should be simple call of:
    //           dst_mem.SetData(dataStore, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(getChildEdgeAt(0)->getMemory(), *dataStore);
}

bool MemoryInput::created() const {
    return getType() == Type::MemoryInput;
}

bool MemoryInput::isExecutable() const {
    return true;
}

void MemoryInput::registerThis(const NodesUnorderedMapPtr & memoryNodes) {
    _memoryNodes = memoryNodes;

    if (auto memoryNodes = _memoryNodes.lock()) {
        auto inputNode = std::static_pointer_cast<MemoryInput>(shared_from_this());
        auto it = memoryNodes->find(getId());
        if (it != memoryNodes->end()) {
            auto outputNode = std::dynamic_pointer_cast<MemoryOutput>(it->second);
            IE_ASSERT(outputNode != nullptr);
            outputNode->setInputNode(inputNode);
        } else {
            memoryNodes->emplace(getId(), std::static_pointer_cast<Node>(inputNode));
        }
    }
}

void MemoryInput::unregisterThis() {
    if (auto memoryNodes = _memoryNodes.lock()) {
        auto it = memoryNodes->find(getId());
        if (it->second.get() == this) {
            memoryNodes->erase(it);
        }
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
