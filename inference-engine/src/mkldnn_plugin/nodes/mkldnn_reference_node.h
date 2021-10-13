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
    MKLDNNReferenceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache, const std::string& errorMessage);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool needShapeInfer() const override {
        if (std::string(opToShapeInfer->get_type_name()) == "Broadcast") {
            return true;
        } else {
            return MKLDNNNode::needShapeInfer();
        }
    }
    bool needPrepareParams() const override { return false; }
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    const std::shared_ptr<ngraph::Node> ngraphOp;
    const std::string additionalErrorMessage;
};

}  // namespace MKLDNNPlugin

