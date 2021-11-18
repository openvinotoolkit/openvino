// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <utils/multidim_map.hpp>
#include <functional>

namespace MKLDNNPlugin {

class MKLDNNColorConvertNode : public MKLDNNNode {
public:
    MKLDNNColorConvertNode(const std::shared_ptr<ngraph::Node>& op,
                           const mkldnn::engine& eng,
                           MKLDNNWeightsSharing::Ptr &cache);
    class Converter;

public:
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    std::vector<VectorDims> shapeInfer() const override;
    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    void initSupportedNV12Impls();

private:
    using ConverterBuilder = std::function<Converter*(MKLDNNNode *)>;
    using SupportedImpls = multidim_map<impl_desc_type, Algorithm, InferenceEngine::Precision::ePrecision, bool, ConverterBuilder>;

    std::unique_ptr<Converter> _impl;
    SupportedImpls _supportedImpls;
};

}  // namespace MKLDNNPlugin
