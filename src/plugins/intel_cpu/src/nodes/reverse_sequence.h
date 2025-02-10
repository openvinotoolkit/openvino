// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class ReverseSequence : public Node {
public:
    ReverseSequence(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    struct ReverseSequenceExecutor {
        ReverseSequenceExecutor(const VectorDims& dataDims,
                                const VectorDims& seqLengthsDims,
                                const VectorDims& dstDims,
                                int batchAxis,
                                int seqAxis);
        ~ReverseSequenceExecutor() = default;

        template <typename T>
        void exec(const MemoryPtr& dataMemPtr, const MemoryPtr& seqLengthsMemPtr, const MemoryPtr& dstMemPtr);

    private:
        const int batchAxis;
        const int seqAxis;
        VectorDims srcStrides;
        size_t workAmountDst;
    };

    using ExecutorPtr = std::shared_ptr<ReverseSequenceExecutor>;
    ExecutorPtr execPtr = nullptr;

    static constexpr size_t REVERSESEQUENCE_DATA = 0;
    static constexpr size_t REVERSESEQUENCE_LENGTHS = 1;

    int seq_axis;
    int batch_axis;

    ov::element::Type lengthsPrecision;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
