// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/permute_kernel.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Transpose : public Node {
public:
    Transpose(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    const InferenceEngine::SizeVector& getOrder() const {
        return order;
    }

    bool isExecutable() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

protected:
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    struct TransposeExecutor {
        TransposeExecutor() = default;
        virtual void exec(Transpose* node, MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr, const int MB) = 0;
        virtual ~TransposeExecutor() = default;
    };
    using executorPtr = std::shared_ptr<TransposeExecutor>;
    executorPtr execPtr = nullptr;

    struct TransposeJitExecutor : public TransposeExecutor {
        TransposeJitExecutor(const PermuteParams& params);
        void exec(Transpose* node, MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr, const int MB) override;

        std::shared_ptr<PermuteKernel> pKernel;
    };

    struct TransposeRefExecutor : public TransposeExecutor {
        TransposeRefExecutor() = default;
        void exec(Transpose* node, MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr, const int MB) override;
    };

    template<typename T> void optimizedExecute(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr);

    InferenceEngine::SizeVector order;
    InferenceEngine::Precision prec;
    bool isOptimized = false;

    const std::vector<std::vector<size_t>> optimizedOrders = {
            std::vector<size_t>{0, 3, 1, 2},
            std::vector<size_t>{0, 4, 1, 2, 3},
            std::vector<size_t>{0, 5, 1, 2, 3, 4},
    };

    PermuteParams params;

    struct TransposeContext {
        Transpose* nodePtr;
        MemoryPtr srcMemPtr;
        MemoryPtr dstMemPtr;
        int MB;
    };

    template<typename T>
    struct TransposeOptimizedEmitter {
        void operator()(TransposeContext& ctx) {
            ctx.nodePtr->optimizedExecute<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
        }
    };

    bool isInputOrderConst = false;

    static constexpr size_t INPUT_DATA_IDX = 0lu;
    static constexpr size_t INPUT_ORDER_IDX = 1lu;

    bool performAsReorder = false;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
