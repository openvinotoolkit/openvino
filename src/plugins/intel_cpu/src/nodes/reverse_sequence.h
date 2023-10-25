// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class ReverseSequence : public Node {
public:
    ReverseSequence(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    struct ReverseSequenceExecutor {
        ReverseSequenceExecutor(const VectorDims& dataDims,
                                const VectorDims& seqLengthsDims,
                                const VectorDims& dstDims,
                                int batchAxis, int seqAxis);
        ~ReverseSequenceExecutor() = default;

        template<typename T>
        void exec(const MemoryPtr& dataMemPtr, const MemoryPtr& seqLengthsMemPtr, const MemoryPtr& dstMemPtr);

    private:
        const int batchAxis;
        const int seqAxis;
        InferenceEngine::SizeVector srcStrides;
        size_t workAmountDst;
    };

    using ExecutorPtr = std::shared_ptr<ReverseSequenceExecutor>;
    ExecutorPtr execPtr = nullptr;

    static constexpr size_t REVERSESEQUENCE_DATA = 0;
    static constexpr size_t REVERSESEQUENCE_LENGTHS = 1;

    int seq_axis;
    int batch_axis;

    InferenceEngine::Precision lengthsPrecision;
    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
