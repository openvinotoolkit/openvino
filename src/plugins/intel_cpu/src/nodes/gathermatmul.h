// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>

#include "cpu_memory.h"
#include "graph_context.h"
#include "node.h"
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

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    enum class Algorithm : uint8_t { GatherMatmulDefault, GatherMatmulCompressed };

    enum InputId : uint8_t {
        DATA = 0,
        WEIGHTS,
        INDICES,
        BIAS,
        WEIGHT_SCALES,
        WEIGHT_ZERO_POINTS,
    };

    Algorithm algorithm = Algorithm::GatherMatmulDefault;
    MemoryArgs memory;
};

}  // namespace ov::intel_cpu::node
