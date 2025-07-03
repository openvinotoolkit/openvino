// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "common/dnnl_executor.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "openvino/core/node.hpp"

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
    using executorPtr = std::shared_ptr<DnnlExecutorLegacy>;
    executorPtr execPtr = nullptr;
    size_t axis = 0;
};

}  // namespace ov::intel_cpu::node
