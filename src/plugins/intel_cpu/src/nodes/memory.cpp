// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory.hpp"

#include <optional>
#include <string>
#include <utility>

#include "common/arbitrary_order_desc_creator.h"
#include "dnnl_extension_utils.h"
#include "dnnl_types.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/cpu_convert.h"
#include "scaled_attn.h"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"
#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov::intel_cpu::node {

namespace {
class MemoryStub : public IMemory {
public:
    class MemoryBlockStub : public IMemoryBlockObserver {
        [[nodiscard]] void* getRawPtr() const noexcept override {
            return nullptr;
        }
        void setExtBuff(void* ptr, size_t size) override {
            // pass
        }
        bool resize(size_t size) override {
            // pass
            return false;
        }
        [[nodiscard]] bool hasExtBuffer() const noexcept override {
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
    MemoryStub(dnnl::engine eng, MemoryDescPtr pMemDesc)
        : m_eng(std::move(eng)),
          m_pMemDesc(std::move(pMemDesc)),
          m_pMemoryBlock(std::make_shared<MemoryBlockStub>()) {}

    [[nodiscard]] const MemoryDesc& getDesc() const override {
        return *m_pMemDesc;
    }

    [[nodiscard]] MemoryDescPtr getDescPtr() const override {
        return m_pMemDesc;
    }

    [[nodiscard]] void* getData() const override {
        OPENVINO_THROW("Unexpected call MemoryStub::getData()");
    }

    [[nodiscard]] size_t getSize() const override {
        return 0;
    }

    [[nodiscard]] const Shape& getShape() const override {
        return m_pMemDesc->getShape();
    }

    [[nodiscard]] const VectorDims& getStaticDims() const override {
        return m_pMemDesc->getShape().getStaticDims();
    }

    void redefineDesc(MemoryDescPtr desc) override {
        m_pMemDesc = desc;
    }

    void load(const IMemory& src, bool ftz, bool bf16saturation) const override {
        OPENVINO_THROW("Unexpected call MemoryStub::load()");
    }

    [[nodiscard]] MemoryBlockPtr getMemoryBlock() const override {
        return m_pMemoryBlock;
    }

    [[nodiscard]] dnnl::memory getPrimitive() const override {
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
}  // namespace

bool MemoryOutputBase::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                            std::string& errorMessage) noexcept {
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

MemoryOutputBase::MemoryOutputBase(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)),
      MemoryNode(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    if (created()) {
        context->getMemoryStatesRegister()->registerOutput(this);
    }
}

MemoryOutputBase::MemoryOutputBase(const std::string& id,
                                   const std::string& name,
                                   const std::string& type,
                                   const Shape& input_shape,
                                   const ov::element::Type& input_prc,
                                   const GraphContext::CPtr& context)
    : Node(type, {input_shape}, {}, {input_prc}, {}, name, context),
      MemoryNode(id) {
    isDynamic = input_shape.isDynamic();
    if (isDynamic) {
        shapeInference = PassThroughShapeInferFactory().makeShapeInfer();
    }
    if (created()) {
        context->getMemoryStatesRegister()->registerOutput(this);
    }
}

MemoryOutputBase::~MemoryOutputBase() {
    if (inputNode) {
        inputNode->deregisterSibling(this);
    }
    context->getMemoryStatesRegister()->remove(this);
}

MemoryInputBase& MemoryOutputBase::getInputNode() {
    CPU_NODE_ASSERT(inputNode, " doesn't have sibling input");
    return *inputNode;
}

void MemoryOutputBase::getSupportedDescriptors() {}

void MemoryOutputBase::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

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
    CPU_NODE_ASSERT(parentPd,
                    parent->getTypeStr(),
                    " ",
                    parent->getName(),
                    "failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    const auto& parentConfig = parentPd->getConfig();
    auto mem_desc = parentConfig.outConfs[parentEdge->getInputNum()].getMemDesc();

    auto selected_pd = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(selected_pd,
                    " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto config = selected_pd->getConfig();

    const bool parentInplaceConflict = parent->inPlaceOutPort(parentEdge->getInputNum()) >= 0;

    // disable inPlace to avoid inPlace conflict and handle memory copy internally (to get room for optimizations)
    if (parentInplaceConflict) {
        config.inConfs.front().inPlace(-1);
    }
    config.inConfs.front().setMemDesc(mem_desc);
    // bypass any checks, we enforce the parent descriptor
    selected_pd->setConfig(config);
}

void MemoryOutputBase::execute(const dnnl::stream& strm) {
    runStatic(strm);
    state->commit();
}

void MemoryOutputBase::executeDynamicImpl(const dnnl::stream& strm) {
    runDynamic(strm);
    state->commit();
}

void MemoryOutputBase::assignState(const MemStatePtr& newState) {
    CPU_NODE_ASSERT(newState, " got null state");
    state = newState;
    assignExtMemory(state->output_mem(), state->internal_desc());
}

bool MemoryOutputBase::neverExecute() const {
    return false;
}

bool MemoryOutputBase::isExecutable() const {
    return true;
}

void MemoryOutputBase::registerInputNode(MemoryInputBase* node) {
    if (inputNode == node) {
        return;
    }
    if (inputNode) {
        inputNode->deregisterSibling(this);
    }
    inputNode = node;
    inputNode->registerOutputNode(this);
}

void MemoryOutputBase::deregisterSibling(MemoryInputBase* node) {
    if (node == inputNode) {
        inputNode = nullptr;
    }
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
    CPU_NODE_ASSERT(selected_pd,
                    " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto parentEdge = getParentEdgeAt(0);  // always only one parent edge

    CPU_NODE_ASSERT(one_of(parentEdge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
                    " Unexpected inplace resolve call to an allocated edge: ",
                    *parentEdge);

    auto memDesc = selected_pd->getConfig().inConfs.front().getMemDesc();
    memBlock = std::make_shared<ProxyMemoryBlock>();
    auto edgeMem = std::make_shared<Memory>(getEngine(), memDesc, memBlock);
    parentEdge->reuse(edgeMem);
}

void MemoryOutput::assignExtMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc) {
    assignedMem = mem;
    CPU_NODE_ASSERT(assignedMem, " assigned state has null memory ptr");

    extMemDesc = memDesc;
    CPU_NODE_ASSERT(extMemDesc, " assigned state has null base mem desc ptr");

    if (!memBlock) {
        return;
    }  // nothing to do, edge memory isn't under control
    auto inpDesc = getBaseMemDescAtInputPort(0);

    if (inpDesc->isCompatible(*extMemDesc)) {
        memBlock->setMemBlockResize(assignedMem->getMemoryBlock());
    } else {
        memBlock->reset();
    }
}

void MemoryOutput::runStatic(dnnl::stream strm) {
    auto inputMem = getSrcMemoryAtPort(0);
    CPU_NODE_ASSERT(assignedMem, " uninitialized assigned memory");

    if (inputMem->getData() != assignedMem->getData()) {
        assignedMem->load(*inputMem, true, false);
    }
}

void MemoryOutput::runDynamic(dnnl::stream strm) {
    // first we have to resize the output memory
    auto inputMem = getSrcMemoryAtPort(0);

    CPU_NODE_ASSERT(assignedMem, " uninitialized assigned memory");

    const auto& newShape = inputMem->getShape();
    const auto& stateShape = assignedMem->getShape();

    if (stateShape.isDynamic() || stateShape.getStaticDims() != newShape.getStaticDims()) {
        CPU_NODE_ASSERT(extMemDesc, " uninitialized assigned memory");
        auto newExternDesc = extMemDesc->cloneWithNewDims(newShape.getStaticDims());
        assignedMem->redefineDesc(newExternDesc);
    }

    if (!newShape.hasZeroDims()) {  // no need to copy data for empty tensor
        runStatic(strm);
    }
}

bool MemoryOutputStub::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                            std::string& errorMessage) noexcept {
    return MemoryOutputBase::isSupportedOperation(op, errorMessage);
}

void MemoryOutputStub::runStatic(dnnl::stream strm) {
    // nothing to do
}

void MemoryOutputStub::runDynamic(dnnl::stream strm) {
    // nothing to do
}

void MemoryOutputStub::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_DOWN)) {
        Node::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(selected_pd,
                    " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto parentEdge = getParentEdgeAt(0);  // always only one parent edge

    CPU_NODE_ASSERT(one_of(parentEdge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
                    " Unexpected inplace resolve call to an allocated edge: ",
                    *parentEdge);

    auto memDesc = selected_pd->getConfig().inConfs.front().getMemDesc();
    // make a fake memory
    auto edgeMem = std::make_shared<MemoryStub>(getEngine(), memDesc);
    parentEdge->reuse(edgeMem);
}

void MemoryOutputStub::assignExtMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc) {
    // nothing to do
}

bool MemoryInputBase::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v3::ReadValue::get_type_info_static(),
                    ov::op::v6::ReadValue::get_type_info_static(),
                    ov::intel_cpu::ReadValueWithSubgraph::get_type_info_static())) {
            errorMessage = "Node is not an instance of ReadValue from the operation set v3 "
                           "or v6, or is not an instance of intel_cpu::ReadValueWithSubgraph";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MemoryInputBase::MemoryInputBase(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& ctx)
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

MemoryInputBase::MemoryInputBase(const std::string& id,
                                 const std::string& name,
                                 const std::string& type,
                                 const Shape& output_shape,
                                 const ov::element::Type& output_prc,
                                 const GraphContext::CPtr& context,
                                 const std::optional<std::vector<Shape>>& input_shape,
                                 const std::optional<std::vector<ov::element::Type>>& input_prc,
                                 MemoryInputBase::mode mode)
    : Input(output_shape, output_prc, name, type, context),
      MemoryStateNode(id) {
    outputShapes.emplace_back(output_shape);
    addOriginalOutputPrecision(output_prc);
    if (input_shape) {
        for (const auto& inp_shape : *input_shape) {
            inputShapes.push_back(inp_shape);
            isDynamic = isDynamic || inp_shape.isDynamic();
        }
        if (isDynamic && !shapeInference) {
            shapeInference = PassThroughShapeInferFactory().makeShapeInfer();
        }
    }
    if (input_prc) {
        for (auto inp_prc : *input_prc) {
            addOriginalInputPrecision(inp_prc);
        }
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
    if (outputNode) {
        outputNode->deregisterSibling(this);
    }
    context->getMemoryStatesRegister()->remove(this);
}

MemoryOutputBase& MemoryInputBase::getOutputNode() {
    CPU_NODE_ASSERT(outputNode, " doesn't have sibling output");
    return *outputNode;
}

void MemoryInputBase::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto precision = getOriginalOutputPrecisionAtPort(0);
    auto&& descCreators = ov::intel_cpu::BlockedDescCreator::getCommonCreators();
    NodeConfig config;

    if (!getParentEdges().empty()) {
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            const auto& inputShape = getInputShapeAtPort(i);
            auto inp_prc = getOriginalInputPrecisionAtPort(i);
            config.inConfs.emplace_back(descCreators.at(LayoutType::ncsp)->createSharedDesc(inp_prc, inputShape));
        }
    }

    const auto& outputShape = getOutputShapeAtPort(0);
    config.outConfs.emplace_back(descCreators.at(LayoutType::ncsp)->createSharedDesc(precision, outputShape),
                                 BlockedMemoryDesc::FULL_MASK,
                                 0);

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MemoryInputBase::registerOutputNode(MemoryOutputBase* node) {
    if (outputNode == node) {
        return;
    }
    if (outputNode) {
        outputNode->deregisterSibling(this);
    }
    outputNode = node;
    outputNode->registerInputNode(this);
}

void MemoryInputBase::deregisterSibling(MemoryOutputBase* node) {
    if (node == outputNode) {
        outputNode = nullptr;
    }
}

bool MemoryInputBase::neverExecute() const {
    return false;
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

void MemoryStatesRegister::registerOutput(MemoryOutputBase* node) {
    OPENVINO_ASSERT(node, "Unexpected null MemoryOutput pointer");
    auto sibling = getMemoryInputByName(node->getId());
    if (sibling != nullptr) {
        node->registerInputNode(sibling);
    }
    memory_outputs[node->getId()] = node;
}

void MemoryStatesRegister::remove(MemoryNode* node) {
    if (nullptr == node) {
        return;
    }
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
    CPU_NODE_ASSERT(newState, " got null state");
    state = newState;
    assignStateHook();
}

void MemoryInputBase::execute(const dnnl::stream& strm) {
    assert(executeHook && "executeHook is not initialized!");
    (this->*executeHook)();
    runStatic(strm);
}

void MemoryInputBase::executeDynamicImpl(const dnnl::stream& strm) {
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

MemoryInput::MemoryInput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& ctx)
    : MemoryInputBase::MemoryInputBase(op, ctx) {
    auto rvWithSubgraph = ov::as_type_ptr<ov::intel_cpu::ReadValueWithSubgraph>(op);
    if (rvWithSubgraph) {
        body = rvWithSubgraph->get_function();
        subGraph = make_unique<ov::intel_cpu::Graph>();
        if (isDynamic) {
            shapeInference = InternalDynShapeInferFactory().makeShapeInfer();
        }
    }
}

MemoryInput::MemoryInput(const std::string& id,
                         const std::string& name,
                         const std::string& type,
                         const Shape& output_shape,
                         const ov::element::Type& output_prc,
                         const GraphContext::CPtr& context,
                         const std::optional<std::vector<Shape>>& input_shape,
                         const std::optional<std::vector<ov::element::Type>>& input_prc,
                         std::shared_ptr<ov::Model> func,
                         mode mode)
    : MemoryInputBase::MemoryInputBase(id, name, type, output_shape, output_prc, context, input_shape, input_prc, mode),
      body(std::move(func)) {
    if (haveSubgraph()) {
        subGraph = make_unique<ov::intel_cpu::Graph>();
        if (isDynamic) {
            shapeInference = InternalDynShapeInferFactory().makeShapeInfer();
        }
    }
}

bool MemoryInput::needInitGraphProcessing() const {
    return !getParentEdges().empty() && getAssignedState()->is_reset_state();
}

void MemoryInput::initOptimalPrimitiveDescriptor() {
    // Mimic the child node memory desc to avoid extra reorder
    static const Type preferredTypes[] = {Type::ScaledDotProductAttention,
                                          Type::MatMul,
                                          Type::FullyConnected,
                                          Type::Convolution,
                                          Type::RNNCell,
                                          Type::RNNSeq,
                                          Type::Subgraph};

    static const Type skipTypes[] = {Type::ShapeOf};

    auto&& childEdges = getChildEdgesAtPort(0);
    EdgePtr childEdge = childEdges.front();

    if (childEdges.size() > 1) {
        // try to prioritize memory desc
        for (auto&& item : childEdges) {
            auto itemType = item->getChild()->getType();
            if (std::any_of(std::begin(skipTypes), std::end(skipTypes), [=](Type type) {
                    return type == itemType;
                })) {
                continue;
            }
            if (std::any_of(std::begin(preferredTypes), std::end(preferredTypes), [=](Type type) {
                    return type == itemType;
                })) {
                childEdge = item;
                break;
            }
        }
    }

    auto child = childEdge->getChild();
    auto childPd = child->getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(childPd,
                    child->getTypeStr(),
                    " ",
                    child->getName(),
                    "failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    const auto& childConfig = childPd->getConfig();
    auto mem_desc = childConfig.inConfs[childEdge->getOutputNum()].getMemDesc();

    auto selectedPd = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(selectedPd,
                    " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto config = selectedPd->getConfig();
    config.outConfs.front().setMemDesc(mem_desc);
    // bypass any checks, we enforce the child descriptor
    selectedPd->setConfig(config);

    if (haveSubgraph()) {
        // Adopt parent configuration, avoid to insert reorder before the MemoryInput.
        std::vector<Input::InputConfig> graphInputConfig;

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto desc = getParentOutputMemDesc(getParentEdgeAt(i));
            graphInputConfig.emplace_back(node::Input::InputConfig{desc, true});
        }

        std::vector<Input::OutputConfig> graphOutputConfig;
        for (auto&& portConfig : config.outConfs) {
            auto desc = portConfig.getMemDesc();
            graphOutputConfig.emplace_back(desc, true);
        }

        // configure the inner graph to get the information about output memory descriptors
        subGraph->Init(body, context, graphInputConfig, graphOutputConfig);
    }
}

// @todo add ascii diagramm for memory mapping / reuse
void MemoryInput::createPrimitive() {
    if (haveSubgraph()) {
        CPU_NODE_ASSERT(getParentEdges().size() == subGraph->inputsNumber(),
                        "The number of node inputs must be equal to the number of inner graph's inputs");

        for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
            auto subgraphInputNode = subGraph->getInputNodeByIndex(i);
            auto subgraphInputMemory = subgraphInputNode->getDstMemoryAtPort(0);
            subgraphMemoryPtrs.push_back(subgraphInputMemory);
        }

        subGraph->Activate();
    }

    MemoryInputBase::createPrimitive();
}

int MemoryInput::registerToAllocationContext(int offset, AllocationContext& context) {
    if (!haveSubgraph()) {
        return Node::registerToAllocationContext(offset, context);
    }

    CPU_NODE_ASSERT(getParentEdges().size() == subGraph->inputsNumber(),
                    "Number of node inputs must be equal the number of inner graph's inputs");

    for (size_t i = 0; i < subGraph->inputsNumber(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto inputEdges = subGraph->getInputNodeByIndex(i)->getChildEdgesAtPort(0);
        for (const auto& inputEdge : inputEdges) {
            CPU_NODE_ASSERT(inputEdge->getStatus() == Edge::Status::Uninitialized,
                            "Expected Uninitialized state for edge: ",
                            *this);
            inputEdge->sharedMemFrom(parentEdge);
        }
    }

    CPU_NODE_ASSERT(subGraph->outputsNumber() <= getChildEdges().size(),
                    "Number of inner graph's outputs must be not greater than number of node outputs");

    for (size_t i = 0; i < subGraph->outputsNumber(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto outputEdge = subGraph->getOutputNodeByIndex(i)->getParentEdgeAt(0);
        CPU_NODE_ASSERT(outputEdge->getStatus() == Edge::Status::Uninitialized,
                        "Expected Uninitialized state for edge: ",
                        *outputEdge);
        outputEdge->sharedMemFrom(childEdge);
    }

    return subGraph->RegisterToAllocationContext(offset, context);
}

void MemoryInput::runDynamic(dnnl::stream strm) {
    auto assignedMem = getAssignedState()->input_mem();

    CPU_NODE_ASSERT(assignedMem, " assigned state has null memory ptr");

    CPU_NODE_ASSERT(memBlock, " has uninitialized memory block.");

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

    MemoryPtr src = assignedMem;  // declare src memory
    if (processInitGraph) {
        if (haveSubgraph()) {
            // put PrepareParams into runDynamic, because init graph is not called each time.
            for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
                // since the external and internal descriptors are compatible, we may pass the descriptor
                subgraphMemoryPtrs[i]->redefineDesc(getSrcMemoryAtPort(i)->getDescPtr());
            }

            subGraph->ResetInferCount();
            subGraph->Infer();
            // depending on the memory sharing solution, we can return here if the memory is substituted from the
            // external graph or override the src pointer with the memory pointer pointing to the subgraph output
            // memory
            CPU_NODE_ASSERT(subGraph->outputsNumber() == 1, "has unexpected number of outputs");
            src = subGraph->getOutputNodeByIndex(0)->getSrcMemoryAtPort(0);

            // since the shape inference(InternalDynShapeInfer, do nothing) is performed, a memory of the extra child
            // edges, attached to the output ports has to be updated after an inference of the inner graph finished
            auto& childEdges = getChildEdges();
            for (size_t j = 1; j < childEdges.size(); j++) {
                auto& childEdge = childEdges[j];
                auto childEdgePtr = childEdge.lock();
                assert(childEdgePtr);
                assert(0 == childEdgePtr->getInputNum());
                childEdgePtr->getMemoryPtr()->redefineDesc(src->getDescPtr());
            }
        } else {
            src = getSrcMemoryAtPort(0);
        }
    }

    // reshape output
    const auto& newDims = src->getStaticDims();
    redefineOutputMemory(0, newDims);

    // copy data when necessary
    if (src->getData() != dst->getData()) {
        dst->load(*src, true, false);
    }
}

void MemoryInput::runStatic(dnnl::stream strm) {
    auto assignedMem = getAssignedState()->input_mem();

    CPU_NODE_ASSERT(assignedMem, "assigned state has null memory ptr");

    const auto& stateDims = assignedMem->getStaticDims();
    const auto& expectedDims = getBaseMemDescAtOutputPort(0)->getShape().getStaticDims();
    CPU_NODE_ASSERT(expectedDims == stateDims,
                    "unexpected state shape: ",
                    vec2str(stateDims),
                    ", while the expected shape: ",
                    vec2str(expectedDims));

    auto internDesc = getBaseMemDescAtOutputPort(0);

    CPU_NODE_ASSERT(memBlock, "has uninitialized memory block.");

    if (internDesc->isCompatible(assignedMem->getDesc())) {
        memBlock->setMemBlock(assignedMem->getMemoryBlock());
    } else {
        memBlock->reset();
    }

    const bool processInitGraph = needInitGraphProcessing();
    MemoryPtr src = assignedMem;  // declare src memory
    if (processInitGraph) {
        if (haveSubgraph()) {
            subGraph->ResetInferCount();
            subGraph->Infer();

            CPU_NODE_ASSERT(subGraph->outputsNumber() == 1, "has unexpected number of outputs");
            src = subGraph->getOutputNodeByIndex(0)->getSrcMemoryAtPort(0);
        } else {
            src = getSrcMemoryAtPort(0);
        }
    }

    // copy data when necessary
    auto dst = getDstMemoryAtPort(0);
    if (src->getData() != dst->getData()) {
        dst->load(*src, true, false);
    }
}

void MemoryInput::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_UP)) {
        ov::intel_cpu::node::Input::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(selected_pd,
                    "failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    auto memDesc = selected_pd->getConfig().outConfs.front().getMemDesc();
    memBlock = std::make_shared<ProxyMemoryBlock>();

    for (auto&& edge : getChildEdgesAtPort(0)) {  // always only one child port
        CPU_NODE_ASSERT(one_of(edge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
                        "Unexpected inplace resolve call to an allocated edge: ",
                        *edge);

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

std::shared_ptr<ov::Model> MemoryInput::getSubGraph() {
    return body;
}

bool MemoryInput::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    return MemoryInputBase::isSupportedOperation(op, errorMessage);
}

MemoryInputSDPA::MemoryInputSDPA(const std::string& id,
                                 const std::string& name,
                                 const std::string& type,
                                 const Shape& output_shape,
                                 const ov::element::Type& output_prc,
                                 const GraphContext::CPtr& context,
                                 const std::optional<std::vector<Shape>>& input_shape,
                                 const std::optional<std::vector<ov::element::Type>>& input_prc,
                                 const std::shared_ptr<ScaledDotProductAttention>& sdpaNode)
    : MemoryInputBase(id, name, type, output_shape, output_prc, context, input_shape, input_prc),
      m_sdpaNode(sdpaNode) {}

void MemoryInputSDPA::createPrimitive() {
    MemoryInputBase::createPrimitive();
    // determine the output node idx
    auto memDesc = getBaseMemDescAtOutputPort(0);
    auto sdpaNode = m_sdpaNode.lock();
    for (auto&& edge : getChildEdgesAtPort(0)) {  // always only one child port
        auto node = edge->getChild();
        if (node == sdpaNode) {
            m_child_port_idx = edge->getOutputNum();
            break;
        }
    }
    CPU_NODE_ASSERT(m_child_port_idx != -1, getName(), " should be connected to SDPA node.");
}

void MemoryInputSDPA::assignStateHook() {
    auto currentState = getAssignedState();
    auto sdpaNode = m_sdpaNode.lock();
    CPU_NODE_ASSERT(sdpaNode, "SDPA node is not available");
    auto sdpaState = std::dynamic_pointer_cast<VariableStateKVcache>(currentState);
    CPU_NODE_ASSERT(sdpaState, "Unexpected state type: ", currentState->get_name());
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
    CPU_NODE_ASSERT(node, "SDPA node is not available");
    auto kv_precision = node->getKVCachePrecision();
    ScaledDotProductAttention::SDPAQuantParam quant_param;
    if (kv_precision == ov::element::u8) {
        const auto& edges_to_past_key = node->getParentEdgeAt(node->getParentEdges().size() - 2);
        const auto& past_key = std::dynamic_pointer_cast<node::MemoryInputBase>(edges_to_past_key->getParent());
        OPENVINO_ASSERT(past_key);
        quant_param = past_key->getId() == state_name ? node->getKeyQuantParam() : node->getValueQuantParam();
    }

    VectorDims order = {2, 0, 1, 3};
    if (!node->getKVCacheOrder().empty()) {
        order = node->getKVCacheOrder();
    }

    auto internal_desc = ArbitraryOrderDescCreator(order).createSharedDesc(kv_precision, outputShapes.at(0));

    return std::make_shared<VariableStateKVcache>(state_name,
                                                  original_desc,
                                                  internal_desc,
                                                  quant_param.isByChannel,
                                                  quant_param.groupSize);
}

void MemoryInputSDPA::runStatic(dnnl::stream strm) {
    // nothing to do
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
        CPU_NODE_ASSERT(stateMem,
                        "Internal state mem id: ",
                        currentState->get_name(),
                        " is empty, node name: ",
                        getName());

        redefineOutputMemory({stateMem->getStaticDims()});
    }
}

void MemoryInputSDPA::resolveInPlaceEdges(Edge::LOOK look) {
    if (getParentEdgeAt(0)) {
        ov::intel_cpu::node::Input::resolveInPlaceEdges(look);
    } else {
        auto memDesc = getBaseMemDescAtOutputPort(0);
        for (auto&& edge : getChildEdgesAtPort(0)) {  // always only one child port
            CPU_NODE_ASSERT(one_of(edge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
                            " Unexpected inplace resolve call to an allocated edge: ",
                            *edge);

            auto edgeMem = std::make_shared<MemoryStub>(getEngine(), memDesc);
            edge->reuse(edgeMem);
        }
    }
}

MemoryInputSingle::MemoryInputSingle(const std::string& id,
                                     const std::string& name,
                                     const std::string& type,
                                     const Shape& output_shape,
                                     const ov::element::Type& output_prc,
                                     const GraphContext::CPtr& context,
                                     const std::optional<std::vector<Shape>>& input_shape,
                                     const std::optional<std::vector<ov::element::Type>>& input_prc,
                                     std::shared_ptr<ov::Model> func)
    : MemoryInput(id,
                  name,
                  type,
                  output_shape,
                  output_prc,
                  context,
                  input_shape,
                  input_prc,
                  std::move(func),
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
        auto result = getDstMemoryAtPort(0);  // only one output port
        auto stateMem = getAssignedState()->output_mem();
        CPU_NODE_ASSERT(stateMem, " state memory has nullptr");
        if (result->getData() != stateMem->getData()) {
            stateMem->load(*result, true, false);
        }
    }
    getAssignedState()->commit();  // since we don't use MemoryOutput, commit must be called to change the reset state
}

void MemoryInputSingle::runDynamic(dnnl::stream strm) {
    MemoryInput::runDynamic(strm);
    if (needInitGraphProcessing()) {
        // since there is no corresponding MemoryOutput node, we need to update the state here
        auto result = getDstMemoryAtPort(0);  // only one output port
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
            stateMem->load(*result, true, false);
        }
    }
    getAssignedState()->commit();  // since we don't use MemoryOutput, commit must be called to change the reset state
}

bool MemoryInputSingle::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                             std::string& errorMessage) noexcept {
    return MemoryInput::isSupportedOperation(op, errorMessage);
}

}  // namespace ov::intel_cpu::node
