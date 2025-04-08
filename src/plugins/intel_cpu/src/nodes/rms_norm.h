// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class RMSNorm : public Node {
public:
    RMSNorm(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::RMS;
    }
    bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    struct Executor {
        virtual void execute(const std::vector<MemoryPtr>& inputs, const MemoryPtr output) = 0;
        virtual ~Executor() = default;
    };

    std::shared_ptr<Executor> m_executor;
    struct RMSNormExecutor;
    friend struct RMSNormKey;

    float m_eps = 0.0f;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
