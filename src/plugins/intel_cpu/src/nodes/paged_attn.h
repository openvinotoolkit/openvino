// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_state.h"
#include "node.h"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils/plain_tensor.hpp"

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
    void gatherConcatPastkvForPagedAttn(const std::vector<MemoryPtr>& inputs);
    ov::element::Type getRuntimePrecision() const override;

    struct Executor {
        virtual void execute(const std::vector<MemoryPtr>& inputs, const MemoryPtr output) = 0;
    };

    std::shared_ptr<Executor> m_executor;
    template <typename T> struct AttentionExecutor;
    friend struct PagedAttentionKey;

    // PagedAttention input index
    static const size_t ID_Q = 0;
    static const size_t ID_K = 1;
    static const size_t ID_V = 2;
    static const size_t ID_KCACHE = 3;
    static const size_t ID_VCACHE = 4;
    static const size_t ID_IS_PROMPT = 5;
    static const size_t ID_SLOT_MAPPING = 6;
    static const size_t ID_MAX_CONTEXT_LEN = 7;
    static const size_t ID_CONTEXT_LENS = 8;
    static const size_t ID_BLOCK_TABLES = 9;
    static const size_t ID_SCALE = 10;
    static const size_t ID_ALIBI_SLOPES = 11;
    static const size_t ID_SLIDING_WINDOW = 12;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
