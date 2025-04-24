// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Pad : public Node {
public:
    Pad(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    void prepareParams() override;
    bool needShapeInfer() const override;
    bool neverExecute() const override;
    bool isExecutable() const override;
    bool needPrepareParams() const override;

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    using VectorIdxs = std::vector<int32_t>;

    enum PadMode { CONSTANT = 0, EDGE = 1, REFLECT = 2, SYMMETRIC = 3 };

    struct PadAttrs {
        PadMode padMode = CONSTANT;
        float padValue = 0.f;
        VectorIdxs padsBegin;
        VectorIdxs padsEnd;
        int beginPadIdx = 0;
        int endPadIdx = 0;
        ov::element::Type prc;
        bool constPadValue = false;
    } attrs;

    struct PadExecutor {
        PadExecutor(const PadAttrs& attrs,
                    const std::vector<MemoryCPtr>& srcMemory,
                    const std::vector<MemoryCPtr>& dstMemory);
        void exec(const MemoryPtr& srcMemPtr, const MemoryPtr& dstMemPtr);
        ~PadExecutor() = default;

    private:
        void padConstant(const MemoryPtr& srcMemPtr, const MemoryPtr& dstMemPtr);
        template <typename T>
        void padConstantCommon(const MemoryPtr& srcMemPtr, const MemoryPtr& dstMemPtr);
        void padConstantZero(const MemoryPtr& srcMemPtr, const MemoryPtr& dstMemPtr);
        void padEdge(const MemoryPtr& srcMemPtr, const MemoryPtr& dstMemPtr);
        void padReflectOrSymmetric(const MemoryPtr& srcMemPtr,
                                   const MemoryPtr& dstMemPtr,
                                   const bool isSymmetric = false);
        void paramsInitialization(const PadAttrs& attrs,
                                  const std::vector<MemoryCPtr>& srcMemory,
                                  const std::vector<MemoryCPtr>& dstMemory);
        void workPartition();
        void innerParamsInitialization();
        inline void getDstIdx(const VectorIdxs& indexes, size_t& dstIdx) const;

        struct PadContext {
            PadExecutor* executor;
            MemoryPtr srcMemPtr;
            MemoryPtr dstMemPtr;
        };

        template <typename T>
        struct PadConstantEmitter {
            void operator()(PadContext& ctx) {
                ctx.executor->padConstantCommon<T>(ctx.srcMemPtr, ctx.dstMemPtr);
            }
        };

        bool zeroInputDimsCase = false;

        struct {
            PadAttrs attrs;
            VectorDims srcDims;
            VectorDims dstDims;
            VectorDims srcODims;
            VectorDims srcStrides;
            VectorDims dstStrides;
            VectorDims srcDimsForReflectOrSymmetric;
            int nThreads = 0;
            size_t nDimsForWork = 0lu;
            size_t workAmount = 0lu;
            size_t lastDstDim = 1lu;
            size_t shift = 0lu;
            size_t dataSize = 1lu;
            size_t innerBeginShift = 0lu;
            size_t innerEndShift = 0lu;
            size_t innerSrcShift = 0lu;
            size_t innerCopySize = 0lu;
            size_t innerBeginPadCount = 0lu;
            size_t innerEndPadCount = 0lu;
            PadMode padMode;
        } params;
    };

    static constexpr size_t DATA_ID = 0lu;
    static constexpr size_t PADS_BEGIN_ID = 1lu;
    static constexpr size_t PADS_END_ID = 2lu;
    static constexpr size_t PAD_VALUE_ID = 3lu;

    bool isPadValueSpecified = false;

    using executorPtr = std::shared_ptr<PadExecutor>;
    executorPtr execPtr = nullptr;
    std::vector<MemoryCPtr> srcMemory;
    std::vector<MemoryCPtr> dstMemory;
    bool shapeHasDataDependency = false;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
