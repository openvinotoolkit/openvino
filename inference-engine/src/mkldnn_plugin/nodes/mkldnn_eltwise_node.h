// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNEltwiseNode : public MKLDNNNode {
public:
    MKLDNNEltwiseNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNEltwiseNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;

    bool isSum();
    bool isUnitScales();
    void initOptimalPrimitiveDescriptor() override;

private:
    static Register<MKLDNNEltwiseNode> reg;
    InferenceEngine::EltwiseLayer::eOperation op;
    std::vector<float> sum_scales;

    template <typename T0, typename T1> void ref_eltwise(int in0, int in1);
};

}  // namespace MKLDNNPlugin

