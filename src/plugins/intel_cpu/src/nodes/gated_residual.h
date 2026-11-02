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
#include "nodes/executors/gated_residual_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/gated_residual.hpp"

namespace ov::intel_cpu::node {

/// @brief CPU node for ov::op::internal::GatedResidual. Accepts f32/bf16/f16 on its data-path ports
/// (all internal accumulation happens in f32) and, optionally, u8/i8/u4/i4 weights with f32
/// decompression scale/zero-point ports. The computation lives in the executor selected through
/// ExecutorFactory<GatedResidualAttrs> (see nodes/executors/dnnl/dnnl_gated_residual_executor.hpp).
class GatedResidual : public Node {
public:
    GatedResidual(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    [[nodiscard]] bool created() const override {
        return getType() == Type::GatedResidual;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    void updateMemoryArgs();

    GatedResidualAttrs m_attrs;
    ExecutorFactoryPtr<GatedResidualAttrs> m_factory;
    ExecutorPtr m_executor;
    MemoryArgs m_memory;
};

}  // namespace ov::intel_cpu::node
