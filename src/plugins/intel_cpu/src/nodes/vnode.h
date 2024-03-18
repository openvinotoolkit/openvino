// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "transformations/cpu_opset/common/op/vnode.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class VNode : public Node {
public:
    VNode(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::VNode;
    }
    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    struct Executor {
        virtual void execute(dnnl::stream strm,
                             intel_cpu::Node * pnode,
                             const intel_cpu::VNode::Config& config) = 0;
    };
    template <typename T>
    struct VNodeExecutorCausalMaskPreprocess;
    intel_cpu::VNode::Config m_config;
    std::shared_ptr<Executor> m_executor;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
