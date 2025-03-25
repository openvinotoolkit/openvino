// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class LoRA : public Node {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    LoRA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    bool created() const override {
        return getType() == Type::LoRA;
    }

    void getSupportedDescriptors() override{};
    void selectOptimalPrimitiveDescriptor() override;
    int registerToAllocationContext(int offset, AllocationContext& context) override;
    void createPrimitive() override;
    void prepareParams() override;
    void execute(const dnnl::stream&) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    std::shared_ptr<const ov::Model> m_body;
    std::vector<MemoryPtr> subgraphMemoryPtrs;
    Graph m_graph;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
