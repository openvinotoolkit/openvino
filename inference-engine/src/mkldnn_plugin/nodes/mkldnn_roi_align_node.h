// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <mkldnn_extension_utils.h>

namespace MKLDNNPlugin {

class MKLDNNROIAlignNode : public MKLDNNNode {
public:
    MKLDNNROIAlignNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int pooledH = 7;
    int pooledW = 7;
    int samplingRatio = 2;
    float spatialScale = 1.0f;
    template <typename inputType, typename outputType>
    void executeSpecified();
    template<typename T>
    struct ROIAlignExecute;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
