// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <cpu_types.h>
#include "ie_algorithm.hpp"
#include "input.h"
#include <node.h>
#include <string>
#include <memory>
#include <map>

namespace ov {
namespace intel_cpu {
namespace node {

class MemoryNode {
protected:
    std::string _id;
    std::weak_ptr<NodesUnorderedMap> _memoryNodes;

public:
    explicit MemoryNode(const std::string & id);
    explicit MemoryNode(const std::shared_ptr<ngraph::Node>& op);
    virtual ~MemoryNode() = default;
    const std::string & getId() const;
    virtual void registerThis(const NodesUnorderedMapPtr & memoryNodes) = 0;
    virtual void unregisterThis() = 0;
};

class MemoryInput : public Input, public MemoryNode {
public:
    MemoryInput(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    ~MemoryInput() override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    bool created() const override;
    bool isExecutable() const override;
    void execute(dnnl::stream strm) override;
    void createPrimitive() override;
    void storeState(const Memory& mem);
    MemoryPtr getStore();
    void registerThis(const NodesUnorderedMapPtr & memoryNodes) override;
    void unregisterThis() override;

 private:
    MemoryPtr dataStore;
};

class MemoryOutput : public Node, public MemoryNode {
public:
    MemoryOutput(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    ~MemoryOutput() override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {}
    void execute(dnnl::stream strm) override;
    bool created() const override;
    void setInputNode(const std::weak_ptr<MemoryInput> & node);
    void registerThis(const NodesUnorderedMapPtr & memoryNodes) override;
    void unregisterThis() override;

private:
    /**
     * @brief keeps reference to input sibling node
     */
    std::weak_ptr<MemoryInput> _inputNode;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
