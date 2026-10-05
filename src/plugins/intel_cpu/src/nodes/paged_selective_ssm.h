// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/paged_selective_ssm_config.hpp"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class PagedSelectiveSSM : public Node {
public:
    PagedSelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::PagedSelectiveSSM;
    }
    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(3);
    }

    void createPrimitive() override;
    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& error_message) noexcept;

private:
    void bindMemoryArguments();
    PagedSelectiveSSMAttrs m_attrs;
    ExecutorFactoryPtr<PagedSelectiveSSMAttrs> m_factory;
    ExecutorPtr m_executor;
    MemoryArgs m_memory;
};

}  // namespace ov::intel_cpu::node
