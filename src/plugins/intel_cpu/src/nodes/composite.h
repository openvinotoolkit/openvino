// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "graph.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Composite : public Node {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    Composite(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    bool created() const override {
        return getType() == Type::SubModel;
    }

    bool needShapeInfer() const override {
        return false;
    }

    bool needPrepareParams() const override {
        return false;
    }

    bool neverExecute() const override {
        return false;
    }

    bool isExecutable() const override {
        return true;
    }

    void getSupportedDescriptors() override{};
    void selectOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute(const dnnl::stream&) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    int registerToAllocationContext(int offset, AllocationContext& context) override;

    const Graph& graph() const {
        return m_graph;
    }

private:
    std::shared_ptr<const ov::Model> m_body;
    Graph m_graph;
    std::shared_ptr<Executor> m_executor;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
