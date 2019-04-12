// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNQuantizeNode : public MKLDNNNode {
public:
    MKLDNNQuantizeNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng);
    ~MKLDNNQuantizeNode() override = default;

    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;


private:
    static Register<MKLDNNQuantizeNode> reg;

    bool canStorePacked;
    int levels;

    std::vector<float> binarizationThresholds;
};

}  // namespace MKLDNNPlugin
