// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <graph.h>
#include <node.h>

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "allocation_context.hpp"
#include "cpu_memory.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/node.hpp"
#include "openvino/op/if.hpp"

namespace ov::intel_cpu::node {

class If : public Node {
public:
    If(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override {}
    int registerToAllocationContext(int offset, AllocationContext& context) override;
    void createPrimitive() override;
    bool created() const override;

    void execute(const dnnl::stream& strm) override;
    bool neverExecute() const override {
        return false;
    }
    bool isExecutable() const override {
        return true;
    }

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool needPrepareParams() const override {
        return false;
    };
    bool needShapeInfer() const override {
        return false;
    }

private:
    void prepareBeforeMappers(bool isThen, const dnnl::engine& eng);
    void prepareAfterMappers(bool isThen, const dnnl::engine& eng);

    static std::deque<MemoryPtr> getToMemories(const Node* node, size_t port);

    struct PortMap {
        int from; /**< Index of external/internal out data */
        int to;   /**< Index of external/internal in data */
    };

    class PortMapHelper {
    public:
        PortMapHelper(MemoryPtr from, std::deque<MemoryPtr> to, const dnnl::engine& eng);
        ~PortMapHelper() = default;
        void execute(const dnnl::stream& strm);

    private:
        void redefineTo();

        MemoryPtr srcMemPtr;
        std::deque<MemoryPtr> dstMemPtrs;
        std::deque<MemoryDescPtr> originalDstMemDescs;

        ptrdiff_t size = 0;
    };

    Graph m_thenGraph;
    Graph m_elseGraph;
    std::vector<std::deque<MemoryPtr>> inputMemThen, inputMemElse;
    std::deque<MemoryPtr> outputMemThen, outputMemElse;

    std::vector<std::shared_ptr<PortMapHelper>> beforeThenMappers, beforeElseMappers, afterThenMappers,
        afterElseMappers;

    std::vector<PortMap> thenInputPortMap, thenOutputPortMap, elseInputPortMap, elseOutputPortMap;

    std::shared_ptr<ov::op::v8::If> m_op;
};

}  // namespace ov::intel_cpu::node
