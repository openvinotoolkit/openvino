// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "edge.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class Split : public Node {
public:
    Split(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    void initOptimalPrimitiveDescriptor() override;

    [[nodiscard]] bool neverExecute() const override;
    [[nodiscard]] bool isExecutable() const override;

    [[nodiscard]] bool needPrepareParams() const override;
    [[nodiscard]] bool needShapeInfer() const override;
    void prepareParams() override;
    void createPrimitive() override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void resolveInPlaceEdges(Edge::LOOK look) override;

private:
    struct SplitExecutor {
        virtual void exec(const uint8_t* srcData, const std::vector<uint8_t*>& dstRawMemPtrs) = 0;
        virtual ~SplitExecutor() = default;
    };
    std::shared_ptr<SplitExecutor> execPtr = nullptr;

    struct SplitOptimizedExecutor : public SplitExecutor {
    public:
        SplitOptimizedExecutor(const BlockedMemoryDescCPtr& inDesc,
                               const std::vector<BlockedMemoryDescCPtr>& outDescs,
                               size_t axis);
        void exec(const uint8_t* srcData, const std::vector<uint8_t*>& dstRawMemPtrs) override;

    private:
        std::vector<size_t> dataSize;
        std::vector<size_t> srcDataOffsets;
        size_t srcDataStride;
        size_t countStrides;
    };

    void optimizedNspc2Ncsp(size_t MB);
    [[nodiscard]] std::vector<uint8_t*> getRawDstMemPtrs() const;

    bool canUseOptimizedNspc2Ncsp = false;

    size_t axis = 1;
    std::vector<std::pair<size_t, MemoryCPtr>> dstMemPtrs;

    size_t INPUTS_NUM = 2;
    bool constSplitLengths = true;
    std::vector<int> splitLengths;
};

}  // namespace ov::intel_cpu::node
