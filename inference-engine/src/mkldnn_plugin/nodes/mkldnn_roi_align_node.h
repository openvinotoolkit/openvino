// Copyright (C) 2020 Intel Corporation
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

enum ROIAlignOpType {
    Max,
    Avg
};

class MKLDNNROIAlignNode : public MKLDNNNode {
public:
    MKLDNNROIAlignNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNROIAlignNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    int pooledH = 7;
    int pooledW = 7;
    int samplingRatio = 2;
    float spatialScale = 1.0f;
    ROIAlignOpType opType = Max;
    template <typename inputType, typename outputType>
    void executeSpecified();
    template<typename T>
    struct ROIAlignExecute;
};
}  // namespace MKLDNNPlugin

