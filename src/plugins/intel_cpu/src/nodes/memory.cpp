// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "dnnl_types.h"
#include "dnnl_extension_utils.h"
#include "memory.hpp"
#include "scaled_attn.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "shape_inference/shape_inference_pass_through.hpp"
#include "common/arbitrary_order_desc_creator.h"

using namespace dnnl;

namespace ov {
namespace intel_cpu {
namespace node {

namespace {
class MemoryStub : public IMemory {
public:
    class MemoryBlockStub : public IMemoryBlockObserver {
        void* getRawPtr() const noexcept override {
            return nullptr;
        }
        void setExtBuff(void* ptr, size_t size) override {
            // pass
        }
        bool resize(size_t size) override {
            // pass
            return false;
        }
        bool hasExtBuffer() const noexcept override {
            // pass
            return false;
        }
        void registerMemory(Memory* memPtr) override {
            // pass
        }
        void unregisterMemory(Memory* memPtr) override {
            // pass
        }
    };

public:
    MemoryStub(const dnnl::engine& eng, const MemoryDescPtr& pMemDesc)
        : m_eng(eng),
          m_pMemDesc(pMemDesc),
          m_pMemoryBlock(std::make_shared<MemoryBlockStub>()) {}

    const MemoryDesc& getDesc() const override {
        return *m_pMemDesc;
    }

    MemoryDescPtr getDescPtr() const override {
        return m_pMemDesc;
    }

    void* getData() const override {
        OPENVINO_THROW("Unexpected call MemoryStub::getData()");
    }

    size_t getSize() const override {
        return 0;
    }

    const Shape& getShape() const override {
        return m_pMemDesc->getShape();
    }

    const VectorDims& getStaticDims() const override {
        return m_pMemDesc->getShape().getStaticDims();
    }

    void redefineDesc(MemoryDescPtr desc) override {
        m_pMemDesc = desc;
    }

    void load(const IMemory& src, bool ftz = true) const override {
        OPENVINO_THROW("Unexpected call MemoryStub::load()");
    }

    MemoryBlockPtr getMemoryBlock() const override {
        return m_pMemoryBlock;
    }

    dnnl::memory getPrimitive() const override {
        OPENVINO_THROW("Unexpected call MemoryStub::getPrimitive()");
    }

    void nullify() override {
        // nothing to do
    }

private:
    dnnl::engine m_eng;
    MemoryDescPtr m_pMemDesc;
    std::shared_ptr<MemoryBlockStub> m_pMemoryBlock;
};
} // namespace

bool MemoryOutputBase::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v3::Assign::get_type_info_static(),
                ov::op::v6::Assign::get_type_info_static())) {
            errorMessage = "Node is not an instance of Assign from the operation set v3 or v6.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MemoryOutputBase::MemoryOutputBase(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) , MemoryNode(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    if (created()) {
        context->getMemoryStatesRegister()->registerOutput(this);
    }
}

MemoryOutputBase::MemoryOutputBase(const std::string id,
                                   const std::string& name,
                                   const std::string& type,
                                   const Shape& input_shape,
                                   const ov::element::Type& input_prc,
                                   const GraphContext::CPtr context) :
    Node(type, {input_shape}, {}, {input_prc}, {}, name, context), MemoryNode(id) {
    isDynamic = input_shape.isDynamic();
    if (isDynamic) {
        shapeInference = PassThroughShapeInferFactory().makeShapeInfer();
    }
    if (created()) {
        context->getMemoryStatesRegister()->registerOutput(this);
    }
}

MemoryOutputBase::~MemoryOutputBase() {
    if (inputNode) { inputNode->deregisterSibling(this); }
    context->getMemoryStatesRegister()->remove(this);
}

MemoryInputBase& MemoryOutputBase::getInputNode() {
    OPENVINO_ASSERT(inputNode, "MemoryOutput ", getName(), " doesn't have sibling input");
    return *inputNode;
}

void MemoryOutputBase::getSupportedDescriptors() {}

void MemoryOutputBase::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto&& shape = getInputShapeAtPort(0);
    auto precision = getOriginalInputPrecisionAtPort(0);
    auto&& descCreators = ov::intel_cpu::BlockedDescCreator::getCommonCreators();

    NodeConfig config;

    PortConfig inPortConfig;
    inPortConfig.inPlace(0);
    inPortConfig.constant(false);
    inPortConfig.setMemDesc(descCreators.at(LayoutType::ncsp)->createSharedDesc(precision, shape));

    config.inConfs.push_back(std::move(inPortConfig));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MemoryOutputBase::initOptimalPrimitiveDescriptor() {
    // Mimic the parent node memory desc to avoid extra reorder
    auto parentEdge = getParentEdgeAt(0);
    auto parent = parentEdge->getParent();
    auto parentPd = parent->getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(parentPd,
        parent->getTypeStr(), " ",
        parent->getName(),
        "failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    const auto& parentConfig = parentPd->getConfig();
    auto mem_desc = parentConfig.outConfs[parentEdge->getInputNum()].getMemDesc();

    auto selected_pd = getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(selected_pd,
        "MemoryOutput ",
        getName(),
        " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto config = selected_pd->getConfig();

    const bool parentInplaceConflict = parent->inPlaceOutPort(parentEdge->getInputNum()) >= 0;

    //disable inPlace to avoid inPlace conflict and handle memory copy internally (to get room for optimizations)
    if (parentInplaceConflict) { config.inConfs.front().inPlace(-1); }
    config.inConfs.front().setMemDesc(mem_desc);
    //bypass any checks, we enforce the parent descriptor
    selected_pd->setConfig(config);
}

void MemoryOutputBase::execute(dnnl::stream strm) {
    runStatic(strm);
    state->commit();
}

void MemoryOutputBase::executeDynamicImpl(dnnl::stream strm) {
    runDynamic(strm);
    state->commit();
}

void MemoryOutputBase::assignState(MemStatePtr newState) {
    OPENVINO_ASSERT(newState, "MemoryOutput ", getName(), " got null state");
    state = newState;
    assignExtMemory(state->output_mem(), state->internal_desc());
}

bool MemoryOutputBase::isExecutable() const {
    return true;
}

void MemoryOutputBase::registerInputNode(MemoryInputBase* node) {
    if (inputNode == node) { return; }
    if (inputNode) { inputNode->deregisterSibling(this); }
    inputNode = node;
    inputNode->registerOutputNode(this);
}

void MemoryOutputBase::deregisterSibling(MemoryInputBase* node) {
    if (node == inputNode) { inputNode = nullptr; }
}

bool MemoryOutput::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    return MemoryOutputBase::isSupportedOperation(op, errorMessage);
}

void MemoryOutput::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_DOWN)) {
        Node::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(selected_pd,
        "MemoryOutput ",
        getName(),
        " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto parentEdge = getParentEdgeAt(0); // always only one parent edge

    OPENVINO_ASSERT(one_of(parentEdge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
        " Unexpected inplace resolve call to an allocated edge: ", parentEdge->name());

    auto memDesc = selected_pd->getConfig().inConfs.front().getMemDesc();
    memBlock = std::make_shared<ProxyMemoryBlock>();
    auto edgeMem = std::make_shared<Memory>(getEngine(), memDesc, memBlock);
    parentEdge->reuse(edgeMem);
}

void MemoryOutput::assignExtMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc) {
    assignedMem = mem;
    OPENVINO_ASSERT(assignedMem,
        "MemoryOutput ",
        getName(),
        " assigned state has null memory ptr");

    extMemDesc = memDesc;
    OPENVINO_ASSERT(extMemDesc,
        "MemoryOutput ",
        getName(),
        " assigned state has null base mem desc ptr");

    if (!memBlock) { return; } //nothing to do, edge memory isn't under control
    auto inpDesc = getBaseMemDescAtInputPort(0);

    if (inpDesc->isCompatible(*extMemDesc)) {
        memBlock->setMemBlockResize(assignedMem->getMemoryBlock());
    } else {
        memBlock->reset();
    }
}

void MemoryOutput::runStatic(dnnl::stream strm)  {
    auto inputMem = getSrcMemoryAtPort(0);
    OPENVINO_ASSERT(assignedMem,
        "MemoryOutput ",
        getName(),
        " uninitialized assigned memory");

    if (inputMem->getData() != assignedMem->getData()) {
        assignedMem->load(*inputMem);
    }
}

void MemoryOutput::runDynamic(dnnl::stream strm) {
    //first we have to resize the output memory
    auto inputMem = getSrcMemoryAtPort(0);

    OPENVINO_ASSERT(assignedMem,
        "MemoryOutput ",
        getName(),
        " uninitialized assigned memory");

    const auto& newShape = inputMem->getShape();
    const auto& stateShape = assignedMem->getShape();

    if (stateShape.isDynamic() || stateShape.getStaticDims() != newShape.getStaticDims()) {
        OPENVINO_ASSERT(extMemDesc,
            "MemoryOutput ",
            getName(),
            " uninitialized assigned memory");
        auto newExternDesc = extMemDesc->cloneWithNewDims(newShape.getStaticDims());
        assignedMem->redefineDesc(newExternDesc);
    }

    if (!newShape.hasZeroDims()) { // no need to copy data for empty tensor
        runStatic(strm);
    }
}

bool MemoryOutputStub::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    return MemoryOutputBase::isSupportedOperation(op, errorMessage);
}

void MemoryOutputStub::runStatic(dnnl::stream strm) {
    //nothing to do
}

void MemoryOutputStub::runDynamic(dnnl::stream strm) {
    //nothing to do
}

void MemoryOutputStub::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_DOWN)) {
        Node::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(selected_pd,
        "MemoryOutput ",
        getName(),
        " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto parentEdge = getParentEdgeAt(0); // always only one parent edge

    OPENVINO_ASSERT(one_of(parentEdge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
        " Unexpected inplace resolve call to an allocated edge: ", parentEdge->name());

    auto memDesc = selected_pd->getConfig().inConfs.front().getMemDesc();
    // make a fake memory
    auto edgeMem = std::make_shared<MemoryStub>(getEngine(), memDesc);
    parentEdge->reuse(edgeMem);
}

void MemoryOutputStub::assignExtMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc) {
    //nothing to do
}

bool MemoryInputBase::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v3::ReadValue::get_type_info_static(),
                ov::op::v6::ReadValue::get_type_info_static())) {
            errorMessage = "Node is not an instance of ReadValue from the operation set v3 or v6.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MemoryInputBase::MemoryInputBase(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr ctx)
    : Input(op, ctx),
      MemoryStateNode(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    if (created()) {
        context->getMemoryStatesRegister()->registerInput(this);
    }
    executeHook = &MemoryInputBase::assignState;
}

MemoryInputBase::MemoryInputBase(const std::string id,
                                 const std::string& name,
                                 const std::string& type,
                                 const Shape& output_shape,
                                 const ov::element::Type& output_prc,
                                 const GraphContext::CPtr context,
                                 const ov::optional<Shape>& input_shape,
                                 const ov::optional<ov::element::Type>& input_prc,
                                 MemoryInputBase::mode mode)
    : Input(output_shape, output_prc, name, type, context),
      MemoryStateNode(id) {
    outputShapes.emplace_back(output_shape);
    addOriginalOutputPrecision(output_prc);
    if (input_shape) {
        inputShapes.push_back(*input_shape);
        isDynamic = isDynamic || input_shape->isDynamic();
        if (isDynamic && !shapeInference) {
            shapeInference = PassThroughShapeInferFactory().makeShapeInfer();
        }
    }
    if (input_prc) {
        addOriginalInputPrecision(*input_prc);
    }
    if (created()) {
        context->getMemoryStatesRegister()->registerInput(this);
    }

    // this important to prevent identifying it as a const when it's on a const path
    constant = ConstantType::StrictNoConst;

    if (mode::read_value_assign == mode) {
        executeHook = &MemoryInputBase::assignState;
    } else if (mode::single_read_value == mode) {
        executeHook = &MemoryInputBase::bypassAssignState;
    } else {
        THROW_CPU_NODE_ERR("Unexpected MemoryInput mode");
    }
}

MemoryInputBase::~MemoryInputBase() {
    if (outputNode) { outputNode->deregisterSibling(this); }
    context->getMemoryStatesRegister()->remove(this);
}

MemoryOutputBase& MemoryInputBase::getOutputNode() {
    OPENVINO_ASSERT(outputNode, "MemoryInput ", getName(), " doesn't have sibling output");
    return *outputNode;
}

void MemoryInputBase::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto precision = getOriginalOutputPrecisionAtPort(0);
    auto&& descCreators = ov::intel_cpu::BlockedDescCreator::getCommonCreators();
    NodeConfig config;

    if (!getParentEdges().empty()) {
        const auto& inputShape = getInputShapeAtPort(0);
        config.inConfs.emplace_back(descCreators.at(LayoutType::ncsp)->createSharedDesc(precision, inputShape));
    }

    const auto& outputShape = getOutputShapeAtPort(0);
    config.outConfs.emplace_back(
        descCreators.at(LayoutType::ncsp)->createSharedDesc(precision, outputShape),
        BlockedMemoryDesc::FULL_MASK,
        0);

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MemoryInputBase::registerOutputNode(MemoryOutputBase* node) {
    if (outputNode == node) { return; }
    if (outputNode) { outputNode->deregisterSibling(this); }
    outputNode = node;
    outputNode->registerInputNode(this);
}

void MemoryInputBase::deregisterSibling(MemoryOutputBase* node) {
    if (node == outputNode) { outputNode = nullptr; }
}

bool MemoryInputBase::isExecutable() const {
    return true;
}

void MemoryStatesRegister::registerInput(MemoryInputBase* node) {
    OPENVINO_ASSERT(node, "Unexpected null MemoryInput pointer");
    // in case of output already registered
    auto sibling = getMemoryOutputByName(node->getId());
    if (sibling != nullptr) {
        node->registerOutputNode(sibling);
    }
    memory_inputs[node->getId()] = node;
}

void MemoryStatesRegister::registerOutput(MemoryOutputBase * node) {
    OPENVINO_ASSERT(node, "Unexpected null MemoryOutput pointer");
    auto sibling = getMemoryInputByName(node->getId());
    if (sibling != nullptr) {
        node->registerInputNode(sibling);
    }
    memory_outputs[node->getId()] = node;
}

void MemoryStatesRegister::remove(MemoryNode* node) {
    if (nullptr == node)
        return;
    ov::util::erase_if(memory_inputs, [&](const InputNodesMap::value_type& it) {
        return it.second == node;
    });
    ov::util::erase_if(memory_outputs, [&](const OutputNodesMap::value_type& it) {
        return it.second == node;
    });
}

MemoryInputBase* MemoryStatesRegister::getMemoryInputByName(const std::string& name) {
    auto it = memory_inputs.find(name);
    if (it == memory_inputs.end()) {
        return nullptr;
    }
    return static_cast<MemoryInputBase*>(it->second);
}

MemoryOutputBase* MemoryStatesRegister::getMemoryOutputByName(const std::string& name) {
    auto it = memory_outputs.find(name);
    if (it == memory_outputs.end()) {
        return nullptr;
    }
    return static_cast<MemoryOutputBase*>(it->second);
}

void MemoryInputBase::assignState(MemStatePtr newState) {
    OPENVINO_ASSERT(newState, "MemoryInput ", getName(), " got null state");
    state = newState;
    assignStateHook();
}

void MemoryInputBase::execute(dnnl::stream strm) {
    assert(executeHook && "executeHook is not initialized!");
    (this->*executeHook)();
    runStatic(strm);
}

void MemoryInputBase::executeDynamicImpl(dnnl::stream strm) {
    assert(executeHook && "executeHook is not initialized!");
    (this->*executeHook)();
    runDynamic(strm);
}

void MemoryInputBase::assignState() {
    getOutputNode().assignState(getAssignedState());
}

void MemoryInputBase::bypassAssignState() {
    // nothing to do
    return;
}

bool MemoryInput::needInitGraphProcessing() const {
    return !getParentEdges().empty() && getAssignedState()->is_reset_state();
}

void MemoryInput::initOptimalPrimitiveDescriptor() {
    // Mimic the child node memory desc to avoid extra reorder
    static const Type preferredTypes[] = {
        Type::ScaledDotProductAttention,
        Type::MatMul,
        Type::FullyConnected,
        Type::Convolution,
        Type::RNNCell,
        Type::RNNSeq,
        Type::Subgraph
    };

    static const Type skipTypes[] = {
        Type::ShapeOf
    };

    auto&& childEdges = getChildEdgesAtPort(0);
    EdgePtr childEdge = childEdges.front();

    if (childEdges.size() > 1) {
        // try to prioritize memory desc
        for (auto&& item : childEdges) {
            auto itemType = item->getChild()->getType();
            if (std::any_of(std::begin(skipTypes), std::end(skipTypes), [=](Type type){ return type == itemType; })) {
                continue;
            }
            if (std::any_of(std::begin(preferredTypes),
                    std::end(preferredTypes), [=](Type type){ return type == itemType; })) {
                childEdge = item;
                break;
            }
        }
    }

    auto child = childEdge->getChild();
    auto childPd = child->getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(childPd,
        child->getTypeStr(), " ",
        child->getName(),
        "failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    const auto& childConfig = childPd->getConfig();
    auto mem_desc = childConfig.inConfs[childEdge->getOutputNum()].getMemDesc();

    auto selectedPd = getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(selectedPd,
        "MemoryInput ",
        getName(),
        " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto config = selectedPd->getConfig();
    config.outConfs.front().setMemDesc(mem_desc);
    //bypass any checks, we enforce the child descriptor
    selectedPd->setConfig(config);
}

void MemoryInput::runDynamic(dnnl::stream strm) {
    auto assignedMem = getAssignedState()->input_mem();

    OPENVINO_ASSERT(assignedMem,
        "MemoryInput ",
        getName(),
        " assigned state has null memory ptr");

    OPENVINO_ASSERT(memBlock,
        "MemoryInput ",
        getName(),
        " has uninitialized memory block.");

    // check whether we can share memory block
    const auto& shape = assignedMem->getShape();
    const bool hasZeroDims = shape.hasZeroDims();
    const bool processInitGraph = needInitGraphProcessing();
    const auto& stateDims = shape.getStaticDims();

    if (hasZeroDims && !processInitGraph) {
        // fast track as we don't really need to share memory and transfer any data for empty tensors
        memBlock->reset();
        redefineOutputMemory(0, stateDims);
        return;
    }

    auto dst = getDstMemoryAtPort(0);
    auto currentOutputDesc = dst->getDescPtr();

    auto internDesc = currentOutputDesc->isDefined() && (currentOutputDesc->getShape().getStaticDims() == stateDims)
                          ? currentOutputDesc
                          : getBaseMemDescAtOutputPort(0)->cloneWithNewDims(stateDims, hasZeroDims);

    if (internDesc->isCompatible(assignedMem->getDesc())) {
        memBlock->setMemBlock(assignedMem->getMemoryBlock());
    } else {
        memBlock->reset();
    }

    //reshape output
    const auto& newDims = processInitGraph ? getSrcMemoryAtPort(0)->getStaticDims() : stateDims;

    redefineOutputMemory(0, newDims);

    //copy data when necessary
    auto src = processInitGraph ? getSrcMemoryAtPort(0) : assignedMem;
    if (src->getData() != dst->getData()) {
        dst->load(*src);
    }
}

void MemoryInput::runStatic(dnnl::stream strm) {
    auto assignedMem = getAssignedState()->input_mem();

    OPENVINO_ASSERT(assignedMem,
        "MemoryInput ",
        getName(),
        " assigned state has null memory ptr");

    const auto& stateDims = assignedMem->getStaticDims();
    const auto& expectedDims = getBaseMemDescAtOutputPort(0)->getShape().getStaticDims();
    OPENVINO_ASSERT(expectedDims == stateDims,
            "MemoryInput ",
            getName(),
            " unexpected state shape: ",
            vec2str(stateDims),
            ", while the expected shape: ",
            vec2str(expectedDims));

    auto internDesc = getBaseMemDescAtOutputPort(0);

    OPENVINO_ASSERT(memBlock,
        "MemoryInput ",
        getName(),
        " has uninitialized memory block.");

    if (internDesc->isCompatible(assignedMem->getDesc())) {
        memBlock->setMemBlock(assignedMem->getMemoryBlock());
    } else {
        memBlock->reset();
    }

    const auto processInitGraph = needInitGraphProcessing();

    //copy data when necessary
    auto src = processInitGraph ? getSrcMemoryAtPort(0) : assignedMem;
    auto dst = getDstMemoryAtPort(0);
    if (src->getData() != dst->getData()) {
        dst->load(*src);
    }
}

void MemoryInput::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_UP)) {
        Node::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(selected_pd,
        "MemoryInput ",
        getName(),
        " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto memDesc = selected_pd->getConfig().outConfs.front().getMemDesc();
    memBlock = std::make_shared<ProxyMemoryBlock>();

    for (auto&& edge : getChildEdgesAtPort(0)) { // always only one child port
        OPENVINO_ASSERT(one_of(edge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
            " Unexpected inplace resolve call to an allocated edge: ", edge->name());

        auto edgeMem = std::make_shared<Memory>(getEngine(), memDesc, memBlock);
        edge->reuse(edgeMem);
    }
}

MemStatePtr MemoryInput::makeState() const {
    // assume ov::Tensor is always dense
    auto original_desc =
        std::make_shared<CpuBlockedMemoryDesc>(getOriginalOutputPrecisionAtPort(0), outputShapes.at(0));

    auto mem_desc = getBaseMemDescAtOutputPort(0);
    const auto& eng = getEngine();

    auto state_name = getId();

    // Remove suffix with pair ID. Internal information.
    auto suffix_idx = state_name.find("/id=");
    if (suffix_idx != std::string::npos) {
        state_name = state_name.substr(0, suffix_idx);
    }

    return std::make_shared<VariableStateDoubleBuffer>(state_name,
        std::make_shared<Memory>(eng, mem_desc),
        std::make_shared<Memory>(eng, mem_desc),
        original_desc);
}

bool MemoryInput::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    return MemoryInputBase::isSupportedOperation(op, errorMessage);
}

MemoryInputSDPA::MemoryInputSDPA(const std::string id,
                                 const std::string& name,
                                 const std::string& type,
                                 const Shape& output_shape,
                                 const ov::element::Type& output_prc,
                                 const GraphContext::CPtr context,
                                 const ov::optional<Shape>& input_shape,
                                 const ov::optional<ov::element::Type>& input_prc,
                                 const std::shared_ptr<ScaledDotProductAttention>& sdpaNode) :
    MemoryInputBase(id, name, type, output_shape, output_prc, context, input_shape, input_prc), m_sdpaNode(sdpaNode) {}

void MemoryInputSDPA::createPrimitive() {
    MemoryInputBase::createPrimitive();
    // determine the output node idx
    auto memDesc = getBaseMemDescAtOutputPort(0);
    auto sdpaNode = m_sdpaNode.lock();
    for (auto&& edge : getChildEdgesAtPort(0)) { // always only one child port
        auto node = edge->getChild();
        if (node == sdpaNode) {
            m_child_port_idx = edge->getOutputNum();
            break;
        }
    }
    OPENVINO_ASSERT(m_child_port_idx != -1, getName(), " should be connected to SDPA node.");
}

void MemoryInputSDPA::assignStateHook() {
    auto currentState = getAssignedState();
    auto sdpaNode = m_sdpaNode.lock();
    OPENVINO_ASSERT(sdpaNode);
    auto sdpaState = std::dynamic_pointer_cast<VariableStateKVcache>(currentState);
    OPENVINO_ASSERT(sdpaState);
    sdpaNode->assignState(sdpaState, m_child_port_idx);
}

MemStatePtr MemoryInputSDPA::makeState() const {
    // assume ov::Tensor is always dense
    auto original_desc =
        std::make_shared<CpuBlockedMemoryDesc>(getOriginalOutputPrecisionAtPort(0), outputShapes.at(0));

    auto mem_desc = getBaseMemDescAtOutputPort(0);

    auto state_name = getId();

    // Remove suffix with pair ID. Internal information.
    auto suffix_idx = state_name.find("/id=");
    if (suffix_idx != std::string::npos) {
        state_name = state_name.substr(0, suffix_idx);
    }

    auto node = m_sdpaNode.lock();
    // retrieve the internal precision and axis order from the SDPA node
    OPENVINO_ASSERT(node);
    auto kv_precision = node->getKVCachePrecision();
    VectorDims order = {2, 0, 1, 3};
    if (!node->getKVCacheOrder().empty())
        order = node->getKVCacheOrder();

    auto internal_desc = ArbitraryOrderDescCreator(order).createSharedDesc(kv_precision, outputShapes.at(0));

    return std::make_shared<VariableStateKVcache>(state_name, original_desc, internal_desc);
}

void MemoryInputSDPA::runStatic(dnnl::stream strm) {
    //nothing to do
}

void MemoryInputSDPA::runDynamic(dnnl::stream strm) {
    auto currentState = getAssignedState();
    if (currentState->is_reset_state()) {
        if (getParentEdges().empty()) {
            auto newShape = MemoryDescUtils::makeDummyShape(getBaseMemDescAtOutputPort(0)->getShape(), 0);
            redefineOutputMemory({newShape.getStaticDims()});
        } else {
            auto inpMem = getSrcMemoryAtPort(0);
            redefineOutputMemory({inpMem->getStaticDims()});
        }
    } else {
        auto stateMem = currentState->input_mem();
        OPENVINO_ASSERT(stateMem,
            "Internal state mem id: ",
            currentState->get_name(),
            " is empty, node name: ",
            getName());

        redefineOutputMemory({stateMem->getStaticDims()});
    }
}

void MemoryInputSDPA::resolveInPlaceEdges(Edge::LOOK look) {
    if (getParentEdgeAt(0)) {
        Node::resolveInPlaceEdges(look);
    } else {
        auto memDesc = getBaseMemDescAtOutputPort(0);
        for (auto&& edge : getChildEdgesAtPort(0)) { // always only one child port
            OPENVINO_ASSERT(one_of(edge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
                " Unexpected inplace resolve call to an allocated edge: ", edge->name());

            auto edgeMem = std::make_shared<MemoryStub>(getEngine(), memDesc);
            edge->reuse(edgeMem);
        }
    }
}

MemoryInputSingle::MemoryInputSingle(const std::string id,
                                     const std::string& name,
                                     const std::string& type,
                                     const Shape& output_shape,
                                     const ov::element::Type& output_prc,
                                     const GraphContext::CPtr context,
                                     const ov::optional<Shape>& input_shape,
                                     const ov::optional<ov::element::Type>& input_prc)
    : MemoryInput(id,
                  name,
                  type,
                  output_shape,
                  output_prc,
                  context,
                  input_shape,
                  input_prc,
                  MemoryInputBase::mode::single_read_value) {}

MemStatePtr MemoryInputSingle::makeState() const {
    // assume ov::Tensor is always dense
    auto original_desc =
        std::make_shared<CpuBlockedMemoryDesc>(getOriginalOutputPrecisionAtPort(0), outputShapes.at(0));

    auto mem_desc = getBaseMemDescAtOutputPort(0);
    const auto& eng = getEngine();

    auto state_name = getId();

    // Remove suffix with pair ID. Internal information.
    auto suffix_idx = state_name.find("/id=");
    if (suffix_idx != std::string::npos) {
        state_name = state_name.substr(0, suffix_idx);
    }

    return std::make_shared<VariableStateSingleBuffer>(state_name,
                                                       std::make_shared<Memory>(eng, mem_desc),
                                                       original_desc);
}

void MemoryInputSingle::runStatic(dnnl::stream strm) {
    MemoryInput::runStatic(strm);
    if (needInitGraphProcessing()) {
        // since there is no corresponding MemoryOutput node, we need to update the state here
        auto result = getDstMemoryAtPort(0); // only one output port
        auto stateMem = getAssignedState()->output_mem();
        CPU_NODE_ASSERT(stateMem, " state memory has nullptr");
        if (result->getData() != stateMem->getData()) {
            stateMem->load(*result);
        }
    }
    getAssignedState()->commit(); // since we don't use MemoryOutput, commit must be called to change the reset state
}

void MemoryInputSingle::runDynamic(dnnl::stream strm) {
    MemoryInput::runDynamic(strm);
    if (needInitGraphProcessing()) {
        // since there is no corresponding MemoryOutput node, we need to update the state here
        auto result = getDstMemoryAtPort(0); // only one output port
        auto state = getAssignedState();
        auto stateMem = state->output_mem();
        CPU_NODE_ASSERT(stateMem, " state memory has nullptr");

        const auto& newShape = result->getShape();
        const auto& stateShape = stateMem->getShape();

        if (stateShape.isDynamic() || stateShape.getStaticDims() != newShape.getStaticDims()) {
            auto extMemDesc = state->internal_desc();
            auto newExternDesc = extMemDesc->cloneWithNewDims(newShape.getStaticDims());
            stateMem->redefineDesc(newExternDesc);
        }

        if (result->getData() != stateMem->getData()) {
            stateMem->load(*result);
        }
    }
    getAssignedState()->commit(); // since we don't use MemoryOutput, commit must be called to change the reset state
}

bool MemoryInputSingle::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                             std::string& errorMessage) noexcept {
    return MemoryInput::isSupportedOperation(op, errorMessage);
}

}  // namespace node
}   // namespace intel_cpu
}   // namespace ov
