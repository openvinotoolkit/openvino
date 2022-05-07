// Copyright (C) 2018-2022 Intel Corporation
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
    ReverseSequence(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool needShapeInfer() const override;
    void executeDynamicImpl(dnnl::stream strm) override;
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override {
        return inputShapesModified();
    };
    void prepareParams() override;

private:
    const size_t REVERSESEQUENCE_DATA = 0;
    const size_t REVERSESEQUENCE_LENGTHS = 1;

    int seq_axis;
    int batch_axis;
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector srcStrides;
    size_t work_amount_dst;

    InferenceEngine::Precision lengthsPrecision;
    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
