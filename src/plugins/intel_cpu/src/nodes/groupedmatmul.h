// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <unordered_map>

#include "config.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/groupedmatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

// Native ov::op::v17::GroupedMatMul / ov::op::internal::GroupedMatMulCompressed. Registered for x64
// only (nodes_factory.cpp); elsewhere the op is lowered to GatherMatmul instead.
class GroupedMatMul : public Node {
public:
    GroupedMatMul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

    bool isExecutable() const override;

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    // Same check without the diagnostic message, for use as a transformation callback
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op) noexcept;

    static bool isSupportedCompressedOperation(const std::shared_ptr<ov::Node>& op,
                                               size_t IC,
                                               size_t OC,
                                               size_t G,
                                               const Config& config) noexcept;
    static ov::element::TypeVector getSupportedCompressedWeightsTypes();
    static ov::element::TypeVector getSupportedCompressedActivationsTypes();

private:
    // Port layout depends on the arity:
    //   2D x 3D: mat_a, mat_b, offsets, [scale, [zero point]]
    //   3D x 3D: mat_a, mat_b,          [scale, [zero point]]
    enum InputId : uint8_t {
        DATA = 0,
        WEIGHTS,
        OFFSETS,  // 2D x 3D form only
    };

    Algorithm algorithm = Algorithm::GroupedMatMulDefault;

    GroupedMatMulAttrs m_attrs;
    ExecutorFactoryPtr<GroupedMatMulAttrs> m_factory;
    ExecutorPtr m_executor;
    MemoryArgs m_memory;
    std::unordered_map<int, int> m_atoi;  // executor arg-id → input port mapping
};

}  // namespace ov::intel_cpu::node
