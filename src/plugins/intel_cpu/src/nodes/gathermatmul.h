// Copyright (C) 2018-2026 Intel Corporation
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
    bool needPrepareParams() const override;
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
    enum class Algorithm : uint8_t { GatherMatmulDefault, GatherMatmulCompressed };

    enum InputId : uint8_t {
        DATA = 0,
        WEIGHTS,
        INDICES,
        BIAS,
        WEIGHT_SCALES,
        WEIGHT_ZERO_POINTS,
    };

    class onednn_matmul;

    using GemvImplPtr = std::shared_ptr<onednn_matmul>;

    Algorithm algorithm = Algorithm::GatherMatmulDefault;
    MemoryArgs memory;
    GemvImplPtr gemv_impl = nullptr;
    GemvImplPtr gemm_impl = nullptr;

    MemoryPtr m_weightsMemory = nullptr;
    MemoryPtr m_scalesMemory = nullptr;
    MemoryPtr m_zpMemory = nullptr;

    MemoryPtr m_tmpInpBuffer = nullptr;
    MemoryDescPtr m_tmpInputDesc = nullptr;
    MemoryDescPtr m_tmpOutputDesc = nullptr;

    bool bf16_amx_mode = false;
};

}  // namespace ov::intel_cpu::node
