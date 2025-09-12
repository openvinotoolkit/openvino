// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

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

    enum PadMode : uint8_t { CONSTANT = 0, EDGE = 1, REFLECT = 2, SYMMETRIC = 3 };

    struct PadAttrs {
        PadMode padMode = CONSTANT;
        float padValue = 0.F;
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
        void padReflectOrSymmetric(const MemoryPtr& srcMemPtr, const MemoryPtr& dstMemPtr, bool isSymmetric = false);
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
            size_t nDimsForWork = 0LU;
            size_t workAmount = 0LU;
            size_t lastDstDim = 1LU;
            size_t shift = 0LU;
            size_t dataSize = 1LU;
            size_t innerBeginShift = 0LU;
            size_t innerEndShift = 0LU;
            size_t innerSrcShift = 0LU;
            size_t innerCopySize = 0LU;
            size_t innerBeginPadCount = 0LU;
            size_t innerEndPadCount = 0LU;
            PadMode padMode = PadMode::CONSTANT;
        } params;
    };

    static constexpr size_t DATA_ID = 0LU;
    static constexpr size_t PADS_BEGIN_ID = 1LU;
    static constexpr size_t PADS_END_ID = 2LU;
    static constexpr size_t PAD_VALUE_ID = 3LU;

    bool isPadValueSpecified = false;

    using executorPtr = std::shared_ptr<PadExecutor>;
    executorPtr execPtr = nullptr;
    std::vector<MemoryCPtr> srcMemory;
    std::vector<MemoryCPtr> dstMemory;
    bool shapeHasDataDependency = false;
};

}  // namespace ov::intel_cpu::node
