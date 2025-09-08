// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "config.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class PagedAttention : public Node {
public:
    PagedAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::PagedAttention;
    }

    // pastkv may have zero dimension
    bool neverExecute() const override {
        return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0) ||
               getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(1) ||
               getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(2);
    }

    // pastkv may have zero dimension
    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(0) && !isInputTensorAtPortEmpty(1) && !isInputTensorAtPortEmpty(2);
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

    static bool isQuantByChannel(Config::CacheQuantMode mode, ov::element::Type precision, bool isKey);

private:
    ov::element::Type getRuntimePrecision() const override;

    std::shared_ptr<ov::Extensions::Cpu::PagedAttentionExecutor> m_executor;
    template <typename T>
    struct AttentionExecutor;
    friend struct PagedAttentionKey;

    bool m_hasScore = false;
};

}  // namespace ov::intel_cpu::node
