// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_state.h"
#include "node.h"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils/plain_tensor.hpp"
#include "kernels/scaled_attn/executor_pa.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class PagedAttention : public Node {
public:
    PagedAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::PagedAttention;
    }
    // pastkv may have zero dimension
    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(0) && !isInputTensorAtPortEmpty(1) && !isInputTensorAtPortEmpty(2);
    }
    bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    ov::element::Type getRuntimePrecision() const override;

    std::shared_ptr<ov::Extensions::Cpu::PagedAttentionExecutor> m_executor;
    template <typename T> struct AttentionExecutor;
    friend struct PagedAttentionKey;

    bool m_hasScore = false;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
