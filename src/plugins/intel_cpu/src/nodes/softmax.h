// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "common/dnnl_executor.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "utils/arch_macros.h"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/executors/x64/softmax_fork_executor.hpp"
#endif

namespace ov::intel_cpu::node {

class SoftMax : public Node {
public:
    SoftMax(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void getSupportedDescriptors() override;
    [[nodiscard]] bool created() const override;
    AttrPtr initPrimitiveAttr() override;
    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    enum class ExecutionPath : std::uint8_t { OneDnn, CustomJit };

#ifdef OPENVINO_ARCH_X86_64
    bool tryInitCustomJitExecutor(const DnnlMemoryDescPtr& inpDesc);
#endif
    void initOneDnnPrimitiveArgs();

    using executorPtr = std::shared_ptr<DnnlExecutorLegacy>;
    executorPtr execPtr = nullptr;
#ifdef OPENVINO_ARCH_X86_64
    std::unique_ptr<ov::intel_cpu::SoftmaxForkExecutor> customJitExec;
#endif
    ExecutionPath executionPath = ExecutionPath::OneDnn;
    size_t axis = 0;
};

}  // namespace ov::intel_cpu::node
