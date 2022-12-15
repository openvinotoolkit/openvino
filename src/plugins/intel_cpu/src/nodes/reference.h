// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Reference : public Node {
public:
    Reference(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache, const std::string& errorMessage);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override { return false; }
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    const std::shared_ptr<ngraph::Node> ngraphOp;
    const std::string additionalErrorMessage;

    bool internalDynamismShapeInfer;
    std::set<int> inputsSupportBF16;
    std::set<int> outputsSupportBF16;

    using TensorEntry = std::tuple<InferenceEngine::Precision, VectorDims, void *, ov::Tensor>;
    using TensorCache = std::map<int, TensorEntry>;

    TensorCache inputTensorCache;
    TensorCache outputTensorCache;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
