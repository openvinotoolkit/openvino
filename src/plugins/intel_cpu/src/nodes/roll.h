// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class Roll : public Node {
public:
    Roll(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    struct RollExecutor {
        RollExecutor(const VectorDims& dataDims,
                     const VectorDims& shiftDims,
                     const VectorDims& axesDims,
                     const VectorDims& dstDims);
        ~RollExecutor() = default;

        template <typename T>
        void exec(const MemoryPtr& dataMemPtr,
                  const MemoryPtr& shiftMemPtr,
                  const MemoryPtr& axesMemPtr,
                  const MemoryPtr& dstMemPtr);

    private:
        const size_t numOfDims;
        const size_t blockSize;
        const size_t numOfIterations;
        const size_t axesLength;
    };

    using ExecutorPtr = std::shared_ptr<RollExecutor>;
    ExecutorPtr execPtr = nullptr;

    static constexpr std::array<size_t, 3> supportedPrecisionSizes{1, 2, 4};
    static constexpr size_t DATA_INDEX = 0UL;
    static constexpr size_t SHIFT_INDEX = 1UL;
    static constexpr size_t AXES_INDEX = 2UL;
};

}  // namespace ov::intel_cpu::node
