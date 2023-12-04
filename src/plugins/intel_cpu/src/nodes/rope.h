// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ie_common.h>
#include <node.h>

#include <memory>
#include <string>
#include <vector>

#include "transformations/cpu_opset/common/op/rope.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class RoPE : public Node {
public:
    RoPE(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::RoPE;
    }
    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    struct Executor {
        virtual void execute(dnnl::stream strm,
                             const RoPENode::Config& config,
                             const std::vector<MemoryPtr>& inputs,
                             const std::vector<MemoryPtr>& outputs) = 0;
    };
    template <typename T>
    struct RoPEExecutorRotateHalf;
    template <typename T>
    struct RoPEExecutorInterleaved;
    template <typename T>
    struct RoPEExecutorChatGLM;
    RoPENode::Config m_config;
    std::shared_ptr<Executor> m_executor;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
