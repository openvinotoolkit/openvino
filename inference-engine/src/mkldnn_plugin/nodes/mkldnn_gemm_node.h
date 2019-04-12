// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGemmNode : public MKLDNNNode {
public:
    MKLDNNGemmNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNGemmNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    int getMaxBatch() override;

private:
    static Register<MKLDNNGemmNode> reg;
    float alpha;
    float beta;
    bool transposeA;
    bool transposeB;

    int xAxis;
    int yAxis;

    bool isThreeInputs;

    std::vector<int> aOffsets;
    std::vector<int> bOffsets;
    std::vector<int> cOffsets;
};

}  // namespace MKLDNNPlugin

