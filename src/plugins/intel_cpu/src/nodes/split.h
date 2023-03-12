// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>

namespace ov {
namespace intel_cpu {
namespace node {

class Split : public Node {
public:
    Split(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool isOptimized() const;
    void initOptimalPrimitiveDescriptor() override;

    void setDynamicBatchLim(int lim) override;
    bool isExecutable() const override;

    bool needPrepareParams() const override;
    bool needShapeInfer() const override;
    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override { execute(strm); }

private:
    struct SplitExecutor {
        virtual void exec(const uint8_t* srcData, const std::vector<uint8_t*>& dstRawMemPtrs,
                          const Dim origBatch, const Dim perInferBatch) = 0;
        virtual ~SplitExecutor() = default;
    };
    std::shared_ptr<SplitExecutor> execPtr = nullptr;

    struct SplitOptimizedExecutor : public SplitExecutor {
        public:
            SplitOptimizedExecutor(BlockedMemoryDescCPtr inDesc, const std::vector<BlockedMemoryDescCPtr> &outDescs, const size_t axis);
            void exec(const uint8_t* srcData, const std::vector<uint8_t*>& dstRawMemPtrs,
                      const Dim origBatch, const Dim perInferBatch) override;

        private:
            std::vector<size_t> dataSize;
            std::vector<size_t> srcDataOffsets;
            size_t srcDataStride;
            size_t countStrides;
    };

    void optimizedNspc2Ncsp(size_t MB);
    std::vector<uint8_t*> getRawDstMemPtrs() const;

    bool canUseOptimizedNspc2Ncsp = false;

    size_t axis = 1;
    std::vector<std::pair<size_t, MemoryCPtr>> dstMemPtrs;

    size_t INPUTS_NUM = 2;
    bool constSplitLengths = true;
    std::vector<int> splitLengths;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
