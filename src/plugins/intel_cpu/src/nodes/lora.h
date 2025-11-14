// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "allocation_context.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class LoRA : public Node {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    LoRA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    bool created() const override {
        return getType() == Type::LoRA;
    }

    void getSupportedDescriptors() override {};
    void selectOptimalPrimitiveDescriptor() override;
    int registerToAllocationContext(int offset, AllocationContext& context) override;
    void createPrimitive() override;
    void prepareParams() override;
    void execute([[maybe_unused]] const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    std::shared_ptr<const ov::Model> m_body;
    std::vector<MemoryPtr> subgraphMemoryPtrs;
    Graph m_graph;
};

}  // namespace ov::intel_cpu::node
