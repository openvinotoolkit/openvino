// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//#include <ie_common.h>
#include <mkldnn_node.h>
//#include <string>

namespace MKLDNNPlugin {

class MKLDNNReferenceNode : public MKLDNNNode {
public:
    MKLDNNReferenceNode(const std::shared_ptr<ngraph::Node>& op,
                        const mkldnn::engine& eng,
                        MKLDNNWeightsSharing::Ptr& cache,
                        const std::string& errorMessage);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    std::vector<std::vector<size_t>> shapeInfer() const override;
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    const std::shared_ptr<ngraph::Node> ngraphOp;
    std::shared_ptr<ngraph::Node> opToShapeInfer;
    const std::string additionalErrorMessage;
};

}  // namespace MKLDNNPlugin
