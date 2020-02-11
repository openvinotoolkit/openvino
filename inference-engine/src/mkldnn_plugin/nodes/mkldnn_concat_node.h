// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <ie_precision.hpp>

namespace MKLDNNPlugin {

class MKLDNNConcatNode : public MKLDNNNode {
public:
    MKLDNNConcatNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNConcatNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void selectOptimalPrimitiveDescriptor() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;

    bool isOptimized() const;

private:
    size_t axis = 0;

    size_t inverseOrder(const InferenceEngine::SizeVector& order, size_t axis);

    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;
};

}  // namespace MKLDNNPlugin

