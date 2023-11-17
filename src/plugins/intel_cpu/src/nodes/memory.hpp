// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <cpu_types.h>
#include "ie_algorithm.hpp"
#include "input.h"
#include <node.h>
#include <memory_state.h>
#include <proxy_mem_mgr.h>
#include <string>
#include <memory>
#include <map>

namespace ov {
namespace intel_cpu {
namespace node {

class MemoryOutput;
class MemoryInputBase;

class MemoryNode {
 public:
    explicit MemoryNode(std::string id) : m_id(id) {}
    explicit MemoryNode(const std::shared_ptr<ov::Node>& op);
    virtual ~MemoryNode() = default;
    std::string getId() const {
        return m_id;
    }

private:
    std::string m_id;
};

/**
 * @brief
 * TODO: ATTENTION: this is a temporary solution, this connection should be keep in graph
 * WARNING: thread_local and holderMutex are not needed if moved into graph
 */
class MemoryNodeVirtualEdge {
public:
    using Holder = std::map<std::string, MemoryNode*>;
    static Holder & getExisted() {
        thread_local static Holder existed;
        return existed;
    }

    static MemoryNode * getByName(Holder& holder, std::string name) {
        auto result = holder.find(name);
        if (result != holder.end()) {
            return result->second;
        }
        return nullptr;
    }

    static Holder* registerOutput(MemoryOutput * node);
    static Holder* registerInput(MemoryInputBase * node);
    static void remove(MemoryNode * node, Holder* holder);
    static std::mutex holderMutex;
};

class MemoryOutput : public Node, public MemoryNode {
public:
    MemoryOutput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    ~MemoryOutput() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override {}
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool created() const override {
        return getType() == Type::MemoryOutput;
    }
    void resolveInPlaceEdges(Edge::LOOK look) override;

    void registerInputNode(MemoryInputBase* node);
    void deregisterSibling(MemoryInputBase* node);

    bool needShapeInfer() const override { return false; }
    bool needPrepareParams() const override { return false; }

    void assignExtMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc);

private:
    MemoryInputBase& getInputNode();

private:
    /**
     * @brief keeps reference to input sibling node
     */
    MemoryInputBase* inputNode = nullptr;
    MemoryPtr assignedMem = nullptr;
    MemoryDescPtr extMemDesc = nullptr; // used for resize
    MemoryNodeVirtualEdge::Holder* holder = nullptr;
    ProxyMemoryMngrPtr memMngr = nullptr;
};

class MemoryInputBase : public Input, public MemoryNode {
public:
    MemoryInputBase(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    MemoryInputBase(const std::string id,
                    const Shape& shape,
                    const ov::element::Type& prc,
                    const std::string& name,
                    const std::string& type,
                    const GraphContext::CPtr context);

    ~MemoryInputBase() override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    bool created() const override {
        return getType() == Type::MemoryInput;
    }

    void execute(dnnl::stream strm) override {/*pass*/}
    void executeDynamicImpl(dnnl::stream strm) override {/*pass*/}

    void createPrimitive() override;

    void resolveInPlaceEdges(Edge::LOOK look) override;

    void registerOutputNode(MemoryOutput* node);
    void deregisterSibling(MemoryOutput* node);

    // May be extracted to some interface when necessary
    virtual void assignState(MemStatePtr newState);
    virtual MemStatePtr makeState() const = 0;

protected:
    MemoryOutput& getOutputNode();

private:
    /**
     * @brief keeps reference to output sibling node
     */
    MemoryOutput* outputNode = nullptr;
    MemoryPtr assignedMem = nullptr;
    MemoryNodeVirtualEdge::Holder* holder = nullptr;
    ProxyMemoryMngrPtr memMngr = nullptr;
};

class MemoryInput : public MemoryInputBase {
public:
    using MemoryInputBase::MemoryInputBase;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;

    MemStatePtr makeState() const override;
};

class MemoryInputSDPA : public MemoryInputBase {
public:
    using MemoryInputBase::MemoryInputBase;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void initSupportedPrimitiveDescriptors() override;

    MemStatePtr makeState() const override;
};
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
