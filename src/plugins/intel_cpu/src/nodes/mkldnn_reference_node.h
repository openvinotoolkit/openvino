// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNReferenceNode : public MKLDNNNode {
public:
    MKLDNNReferenceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache, const std::string& errorMessage);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    std::vector<VectorDims> shapeInfer() const override;
    bool needShapeInfer() const override;
    bool needPrepareParams() const override { return false; }
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    const std::shared_ptr<ngraph::Node> ngraphOp;
    const std::string additionalErrorMessage;
};

}  // namespace MKLDNNPlugin

