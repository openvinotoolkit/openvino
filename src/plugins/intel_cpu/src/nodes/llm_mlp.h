// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"

namespace ov::intel_cpu::node {

class LLMMLP : public Node {
public:
    LLMMLP(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::LLMMLP;
    }
    bool needPrepareParams() const override {
        return false;
    }
    void createPrimitive() override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                     std::string& errorMessage,
                                     uint64_t fcDynamicQuantizationGroupSize = 0) noexcept;

private:
    struct ExecutorBase {
        virtual void execute() = 0;
        virtual ~ExecutorBase() = default;
    };
    std::shared_ptr<ExecutorBase> m_executor;
    template <typename T>
    struct Executor;
    LLMMLPNode::Config m_mlp_config{};
};

}  // namespace ov::intel_cpu::node
