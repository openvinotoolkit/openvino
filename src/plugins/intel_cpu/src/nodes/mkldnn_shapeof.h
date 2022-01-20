// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <mkldnn_extension_utils.h>

namespace MKLDNNPlugin {

class MKLDNNShapeOfNode : public MKLDNNNode {
public:
    MKLDNNShapeOfNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool needPrepareParams() const override {return false;};
    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }
    std::vector<VectorDims> shapeInfer() const override {
        return {VectorDims{getParentEdgesAtPort(0)[0]->getMemory().getStaticDims().size()}};
    }

    bool isExecutable() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
