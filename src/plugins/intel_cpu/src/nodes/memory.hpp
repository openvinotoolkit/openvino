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
class MemoryInput;

class MemoryNode {
    std::string _id;
 public:
    explicit MemoryNode(std::string id) : _id(id) {}
    explicit MemoryNode(const std::shared_ptr<ov::Node>& op);
    virtual ~MemoryNode() = default;
    std::string getId() {
        return _id;
    }
    virtual void registerInputNode(MemoryInput*) = 0;
    virtual void registerOutputNode(MemoryOutput*) = 0;
    virtual void deregisterSibling(MemoryNode*) = 0;
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
    static Holder* registerInput(MemoryInput * node);
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

    void registerInputNode(MemoryInput* node) override;
    void registerOutputNode(MemoryOutput* node) override {
        OPENVINO_THROW("MemoryOutput node has no MemoryOutput type sibling!");
    }

    void deregisterSibling(MemoryNode* node) override;

    bool needShapeInfer() const override { return false; }
    bool needPrepareParams() const override { return false; }

    void assignMemory(const MemoryPtr& mem, const MemoryDescPtr& memDesc);

private:
    MemoryInput& getInputNode();

private:
    /**
     * @brief keeps reference to input sibling node
     */
    MemoryInput* inputNode = nullptr;
    MemoryPtr assignedMem = nullptr;
    MemoryDescPtr extMemDesc = nullptr; // used for resize
    MemoryNodeVirtualEdge::Holder* holder = nullptr;
    ProxyMemoryMngrPtr memMngr = nullptr;
};

class MemoryInput : public Input, public MemoryNode {
public:
    MemoryInput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    ~MemoryInput() override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    bool created() const override {
        return getType() == Type::MemoryInput;
    }
    bool isExecutable() const override {
        return true;
    }

    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;

    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override { execute(strm); };

    void createPrimitive() override;

    void resolveInPlaceEdges(Edge::LOOK look) override;

    void registerInputNode(MemoryInput* node) override {
        OPENVINO_THROW("MemoryInput node has no MemoryInput type sibling!");
    }

    void registerOutputNode(MemoryOutput* node) override;
    void deregisterSibling(MemoryNode* node) override;

    void assignState(MemStatePtr newState);
    std::function<MemoryPtr(void)> memoryBuilder() const;

private:
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

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
