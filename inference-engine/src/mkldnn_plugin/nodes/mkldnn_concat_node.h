// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <ie_precision.hpp>

namespace MKLDNNPlugin {

typedef struct {
    const uint8_t* src;
    uint8_t* dst;
    size_t src_stride;
    size_t dst_stride;
    size_t byte_size;
    size_t iters;
    size_t input_size;
} CopyMemTask;

class MKLDNNConcatNode : public MKLDNNNode {
public:
    MKLDNNConcatNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
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

    std::vector<CopyMemTask> _tasks;
};

}  // namespace MKLDNNPlugin

