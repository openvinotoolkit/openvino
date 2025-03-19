// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "kernels/x64/mlp_kernel.hpp"
#endif

namespace ov {
namespace intel_cpu {
namespace node {

class QKVProjection : public Node {
public:
    QKVProjection(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::QKVProjection;
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
                                     int concurrency = 0,
                                     uint64_t fcDynamicQuantizationGroupSize = 0) noexcept;

private:
    struct ExecutorBase {
        virtual void execute() = 0;
        virtual ~ExecutorBase() = default;
    };
    std::shared_ptr<ExecutorBase> m_executor;
    template <typename T>
    struct Executor;

    QKVProjectionNode::Config m_config;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
