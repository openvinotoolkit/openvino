// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "common/permute_kernel.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class DepthToSpace : public Node {
public:
    DepthToSpace(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    void prepareParams() override;

    enum Mode : uint8_t { BLOCKS_FIRST = 0, DEPTH_FIRST = 1 };
    struct DepthToSpaceAttrs {
        LayoutType layoutType = LayoutType::nspc;
        Mode mode = BLOCKS_FIRST;
        size_t blockSize = 0LU;
        size_t blockStep = 0LU;
        size_t dataSize = 1LU;
        size_t nSpatialDims = 0LU;
        VectorDims srcBlockedDims;
        [[nodiscard]] size_t hash() const;
        bool operator==(const DepthToSpaceAttrs& rhs) const;
    };

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    DepthToSpaceAttrs attrs;
    struct DepthToSpaceExecutor {
        DepthToSpaceExecutor(const DepthToSpaceAttrs& attrs);
        void exec(const MemoryPtr& srcMemPtr, const MemoryPtr& dstMemPtr, int MB);
        ~DepthToSpaceExecutor() = default;

    private:
        std::unique_ptr<PermuteKernel> permuteKernel;
    };
    using executorPtr = std::shared_ptr<DepthToSpaceExecutor>;
    executorPtr execPtr = nullptr;
};

}  // namespace ov::intel_cpu::node
