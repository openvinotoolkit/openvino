// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNPadNode : public MKLDNNNode {
public:
    MKLDNNPadNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    void prepareParams() override;

    bool isExecutable() const override;

protected:
    std::vector<VectorDims> shapeInfer() const override;
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    enum PadMode {
        CONSTANT = 0,
        EDGE = 1,
        REFLECT = 2,
        SYMMETRIC = 3
    };

    struct PadAttrs {
        PadMode padMode = CONSTANT;
        float padValue = 0.f;
        std::vector<unsigned int> padsBegin;
        std::vector<unsigned int> padsEnd;
        int beginPadIdx = 0;
        int endPadIdx = 0;
        InferenceEngine::Precision prc;
    } attrs;

    struct PadExecutor {
        PadExecutor(const PadAttrs& params, const VectorDims& srcDims, const VectorDims& dstDims);
        void exec(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr);
        ~PadExecutor() = default;

    private:
        void padConstant(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr);
        template<typename T> void padConstantCommon(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr);
        void padConstantZero(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr);
        void padEdge(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr);
        void padReflectOrSymmetric(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr, const bool isSymmetric = false);

        inline void getDstIdx(const VectorDims& indexes, size_t& dstIdx) const;

        struct PadContext {
            PadExecutor* executor;
            MKLDNNMemoryPtr srcMemPtr;
            MKLDNNMemoryPtr dstMemPtr;
        };

        template<typename T>
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
};

}  // namespace MKLDNNPlugin
