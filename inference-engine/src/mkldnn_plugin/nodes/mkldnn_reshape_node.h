// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <memory>

namespace MKLDNNPlugin {

class MKLDNNReshapeNode : public MKLDNNNode {
public:
    MKLDNNReshapeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    MKLDNNReshapeNode(const std::string& name,
                    const Shape& inDims,
                    const Shape& outDims,
                    InferenceEngine::Precision precision,
                    const mkldnn::engine& eng,
                    MKLDNNWeightsSharing::Ptr &wCache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override { return false; }
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    enum class shapeTypeNode {
        RESHAPE,
        SQUEEZE,
        UNSQUEEZE
    } opType;

    VectorDims outputShape;
    mutable std::vector<int> lastSecondInputValues;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin

