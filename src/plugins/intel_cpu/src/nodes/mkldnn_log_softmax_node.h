// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNLogSoftmaxNode : public MKLDNNNode {
public:
    MKLDNNLogSoftmaxNode(const std::shared_ptr<ngraph::Node>& op,
        const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int axis;
    size_t reducedAxisSize = 0;
    size_t reducedAxisStride = 1;
    size_t axisStep = 1;
    bool isLastDim = false;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
