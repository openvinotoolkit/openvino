// Copyright (C) 2018-2019 Intel Corporation
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
    MKLDNNQuantizeNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNQuantizeNode() override = default;

    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;

    const float* getBinarizationTresholdsPtr() {
        if (!initialized)
            initValues();
        return &binarizationThresholds[0];
    }

    size_t getBinarizationTresholdsSize() {
        if (!initialized)
            initValues();
        return binarizationThresholds.size();
    }

    const float* getBinarizationOutputMaskPtr() {
        if (!initialized)
            initValues();
        return reinterpret_cast<float*>(&binarizationOutputMask[0]);
    }

    size_t getBinarizationOutputMaskSize() {
        if (!initialized)
            initValues();
        return binarizationOutputMask.size();
    }

    bool isPackedStore() {
        if (!initialized)
            initValues();
        return canStorePacked;
    }

private:
    void initValues();

    bool initialized = false;
    bool canStorePacked = false;
    int levels = -1;

    std::vector<float> binarizationThresholds;
    std::vector<uint32_t> binarizationOutputMask;
};

}  // namespace MKLDNNPlugin
