// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNConcatNode : public MKLDNNNode {
public:
    MKLDNNConcatNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNConcatNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void selectOptimalPrimitiveDescriptor() override;
    bool created() const override;

    bool isOptimized() const;

private:
    static Register<MKLDNNConcatNode> reg;
    size_t axis = 0;

    size_t inverseOrder(const InferenceEngine::SizeVector& order, size_t axis);
};

}  // namespace MKLDNNPlugin

