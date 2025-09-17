// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class MatMul : public Node {
public:
    MatMul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    void initSupportedPrimitiveDescriptors() override;
    [[nodiscard]] bool canFuse(const NodePtr& node) const override;
    [[nodiscard]] bool created() const override;

    [[nodiscard]] ov::element::Type getRuntimePrecision() const override;
    [[nodiscard]] const std::vector<impl_desc_type>& getDefaultImplPriority() override;

    [[nodiscard]] int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() - 1;
    }

    void createPrimitive() override;
    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    [[nodiscard]] bool canBeExecutedInInt8() const override;

    [[nodiscard]] bool neverExecute() const override;
    [[nodiscard]] bool isExecutable() const override;

private:
    std::tuple<VecMemoryDescs, MemoryDescPtr> initMemoryDescriptors(ov::element::Type dstType) const;
    ExecutorFactoryPtr<MatMulAttrs> createExecutorFactory(const MemoryDescArgs& descs, const MatMulAttrs& attrs);

    // Attributes structure for MatMul node
    MatMulAttrs m_attrs;

    // Factory for creating executors
    ExecutorFactoryPtr<MatMulAttrs> m_factory;

    // Executor instance
    ExecutorPtr m_executor;

    // Memory arguments mapping
    MemoryArgs m_memory;

    // Argument to input port mapping
    std::unordered_map<int, int> m_atoi;
};

}  // namespace ov::intel_cpu::node
