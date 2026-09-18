// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <unordered_map>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class GatherMatmul : public Node {
public:
    GatherMatmul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

    bool isExecutable() const override;

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    static bool isSupportedCompressedOperation(const std::shared_ptr<ov::Node>& op,
                                               size_t IC,
                                               size_t OC,
                                               size_t G,
                                               const Config& config) noexcept;
    static ov::element::TypeVector getSupportedCompressedWeightsTypes(bool apply_fp8 = false);
    static ov::element::TypeVector getSupportedCompressedActivationsTypes();

private:
    enum InputId : uint8_t {
        DATA = 0,
        WEIGHTS,
        INDICES,
        BIAS,
        WEIGHT_SCALES,
        WEIGHT_ZERO_POINTS,
    };

    Algorithm algorithm = Algorithm::GatherMatmulDefault;

    GatherMatmulAttrs m_attrs;
    ExecutorFactoryPtr<GatherMatmulAttrs> m_factory;
    ExecutorPtr m_executor;
    MemoryArgs m_memory;
    std::unordered_map<int, int> m_atoi;  // executor arg-id → input port mapping
};

}  // namespace ov::intel_cpu::node