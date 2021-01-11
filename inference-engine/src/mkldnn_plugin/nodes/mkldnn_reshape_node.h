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
    MKLDNNReshapeNode(const InferenceEngine::SizeVector &inDims, const InferenceEngine::SizeVector &outDims, const InferenceEngine::Precision &prc,
                      const std::string &nodeName, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNReshapeNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
};

}  // namespace MKLDNNPlugin

