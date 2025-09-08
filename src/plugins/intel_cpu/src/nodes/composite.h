// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "allocation_context.hpp"
#include "cpu_types.h"
#include "graph.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

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

    void getSupportedDescriptors() override {};
    void selectOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute([[maybe_unused]] const dnnl::stream& strm) override;
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

}  // namespace ov::intel_cpu::node
