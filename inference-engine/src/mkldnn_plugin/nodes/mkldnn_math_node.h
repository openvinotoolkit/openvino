// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNMathNode : public MKLDNNNode {
public:
    MKLDNNMathNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    static std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, MKLDNNMathNode& node)>> initializers;

    float alpha = 0.0f;
    float beta = 0.0f;
    float gamma = 0.0f;

    std::vector<DataConfigurator> inDataConf;
    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
