// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/gated_residual_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_residual.hpp"

namespace ov::intel_cpu::node {

/// @brief CPU node for ov::op::internal::GatedResidual (x86-64 only). Data ports are f32/bf16/f16; compressed
/// weights are u8/i8/u4/i4. The computation lives in the executor from ExecutorFactory<GatedResidualAttrs>.
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
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    // Whether the op in its compressed form is worth running: isSupportedOperation plus the limits of the oneDNN
    // weights decompression the projections run through, for each of its weights.
    static bool isSupportedCompressedOperation(const std::shared_ptr<const ov::Node>& op) noexcept;
    // A subset of the FC types: the projections run as FC, and the op handles 4/8-bit integer weights only.
    static ov::element::TypeVector getSupportedCompressedWeightsTypes();

private:
    void bindMemoryArgs();

    GatedResidualAttrs m_attrs;
    ExecutorFactoryPtr<GatedResidualAttrs> m_factory;
    ExecutorPtr m_executor;
    MemoryArgs m_memory;
};

}  // namespace ov::intel_cpu::node
