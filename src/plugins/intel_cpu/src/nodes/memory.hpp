// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <graph.h>

#include <map>
#include <optional>

#include "input.h"
#include "memory_state_base.h"
#include "proxy_mem_blk.h"

namespace ov::intel_cpu::node {

class MemoryOutputBase;
class MemoryInputBase;
class ScaledDotProductAttention;

class MemoryStatesRegister {
public:
    using InputNodesMap = std::unordered_map<std::string, MemoryStateNode*>;
    using OutputNodesMap = std::unordered_map<std::string, MemoryNode*>;

public:
    void registerOutput(MemoryOutputBase* node);
    void registerInput(MemoryInputBase* node);
    void remove(MemoryNode* node);

    const InputNodesMap& getMemoryStates() {
        return memory_inputs;
    }

private:
    MemoryInputBase* getMemoryInputByName(const std::string& name);
    MemoryOutputBase* getMemoryOutputByName(const std::string& name);

private:
    InputNodesMap memory_inputs;
    OutputNodesMap memory_outputs;
};

class MemoryOutputBase : public Node, public MemoryNode {
public:
    MemoryOutputBase(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    MemoryOutputBase(const std::string& id,
                     const std::string& name,
                     const std::string& type,
                     const Shape& input_shape,
                     const ov::element::Type& input_prc,
                     const GraphContext::CPtr& context);

    ~MemoryOutputBase() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override {}
    bool created() const override {
        return getType() == Type::MemoryOutput;
    }

    void execute(const dnnl::stream& strm) override final;
    void executeDynamicImpl(const dnnl::stream& strm) override final;

    bool isExecutable() const override final;
    bool neverExecute() const override final;

    void registerInputNode(MemoryInputBase* node);
    void deregisterSibling(MemoryInputBase* node);

    bool needShapeInfer() const override {
        return false;
    }
    bool needPrepareParams() const override {
        return false;
    }

    void assignState(const MemStatePtr& newState);

protected:
    virtual void runStatic(dnnl::stream strm) = 0;
    virtual void runDynamic(dnnl::stream strm) = 0;
    virtual void assignExtMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc) = 0;
    MemoryInputBase& getInputNode();

private:
    /**
     * @brief keeps reference to input sibling node
     */
    MemoryInputBase* inputNode = nullptr;
    MemStatePtr state = nullptr;  // keep reference to call commit()
};

class MemoryOutput : public MemoryOutputBase {
public:
    using MemoryOutputBase::MemoryOutputBase;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void resolveInPlaceEdges(Edge::LOOK look) override;

protected:
    void runStatic(dnnl::stream strm) override;
    void runDynamic(dnnl::stream strm) override;
    void assignExtMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc) override;

private:
    MemoryPtr assignedMem = nullptr;
    MemoryDescPtr extMemDesc = nullptr;  // used for resize
    ProxyMemoryBlockPtr memBlock = nullptr;
};

class MemoryOutputStub : public MemoryOutputBase {
public:
    using MemoryOutputBase::MemoryOutputBase;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void resolveInPlaceEdges(Edge::LOOK look) override;

protected:
    void runStatic(dnnl::stream strm) override;
    void runDynamic(dnnl::stream strm) override;
    void assignExtMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc) override;
};

class MemoryInputBase : public Input, public MemoryStateNode {
public:
    enum class mode { read_value_assign, single_read_value };

public:
    MemoryInputBase(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    ~MemoryInputBase() override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    bool created() const override {
        return getType() == Type::MemoryInput;
    }

    void initSupportedPrimitiveDescriptors() override;

    void execute(const dnnl::stream& strm) override final;
    void executeDynamicImpl(const dnnl::stream& strm) override final;
    bool needShapeInfer() const override {
        return false;
    }
    bool needPrepareParams() const override {
        return false;
    }
    bool neverExecute() const override final;
    bool isExecutable() const override final;

    void registerOutputNode(MemoryOutputBase* node);
    void deregisterSibling(MemoryOutputBase* node);

    MemoryOutputBase& getOutputNode();
    void assignState(MemStatePtr newState) override final;

protected:
    MemoryInputBase(const std::string& id,
                    const std::string& name,
                    const std::string& type,
                    const Shape& output_shape,
                    const ov::element::Type& output_prc,
                    const GraphContext::CPtr& context,
                    const std::optional<std::vector<Shape>>& input_shape,
                    const std::optional<std::vector<ov::element::Type>>& input_prc,
                    mode mode = mode::read_value_assign);

protected:
    virtual void runStatic(dnnl::stream strm) = 0;
    virtual void runDynamic(dnnl::stream strm) = 0;
    virtual void assignStateHook() = 0;
    MemStatePtr getAssignedState() const {
        return state;
    }

private:
    using executeHookPtr = void (MemoryInputBase::*)();

private:
    void assignState();
    void bypassAssignState();

private:
    /**
     * @brief keeps reference to output sibling node
     */
    MemoryOutputBase* outputNode = nullptr;
    MemStatePtr state = nullptr;
    executeHookPtr executeHook;
};

class MemoryInput : public MemoryInputBase {
public:
    MemoryInput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& ctx);
    MemoryInput(const std::string& id,
                const std::string& name,
                const std::string& type,
                const Shape& output_shape,
                const ov::element::Type& output_prc,
                const GraphContext::CPtr& context,
                const std::optional<std::vector<Shape>>& input_shape,
                const std::optional<std::vector<ov::element::Type>>& input_prc,
                std::shared_ptr<ov::Model> func = nullptr,
                mode mode = mode::read_value_assign);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void initOptimalPrimitiveDescriptor() override;

    void resolveInPlaceEdges(Edge::LOOK look) override;

    int registerToAllocationContext(int offset, AllocationContext& context) override;

    void createPrimitive() override;

    MemStatePtr makeState() const override;

    std::shared_ptr<ov::Model> getSubGraph();

protected:
    bool needInitGraphProcessing() const;
    void runStatic(dnnl::stream strm) override;
    void runDynamic(dnnl::stream strm) override;

private:
    void assignStateHook() override { /*pass*/
    }

    bool haveSubgraph() const {
        return body != nullptr;
    }

private:
    std::shared_ptr<ov::Model> body = nullptr;
    std::unique_ptr<ov::intel_cpu::Graph> subGraph = nullptr;
    std::vector<MemoryPtr> subgraphMemoryPtrs;

    ProxyMemoryBlockPtr memBlock = nullptr;
};

class MemoryInputSingle : public MemoryInput {
public:
    MemoryInputSingle(const std::string& id,
                      const std::string& name,
                      const std::string& type,
                      const Shape& output_shape,
                      const ov::element::Type& output_prc,
                      const GraphContext::CPtr& context,
                      const std::optional<std::vector<Shape>>& input_shape,
                      const std::optional<std::vector<ov::element::Type>>& input_prc,
                      std::shared_ptr<ov::Model> func);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    MemStatePtr makeState() const override;

private:
    void runStatic(dnnl::stream strm) override;
    void runDynamic(dnnl::stream strm) override;
};

class MemoryInputSDPA : public MemoryInputBase {
public:
    MemoryInputSDPA(const std::string& id,
                    const std::string& name,
                    const std::string& type,
                    const Shape& output_shape,
                    const ov::element::Type& output_prc,
                    const GraphContext::CPtr& context,
                    const std::optional<std::vector<Shape>>& input_shape,
                    const std::optional<std::vector<ov::element::Type>>& input_prc,
                    const std::shared_ptr<ScaledDotProductAttention>& sdpaNode);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void createPrimitive() override;
    void resolveInPlaceEdges(Edge::LOOK look) override;

    MemStatePtr makeState() const override;

private:
    void assignStateHook() override;
    void runStatic(dnnl::stream strm) override;
    void runDynamic(dnnl::stream strm) override;

private:
    std::weak_ptr<ScaledDotProductAttention> m_sdpaNode;
    int m_child_port_idx = -1;
};
}  // namespace ov::intel_cpu::node
