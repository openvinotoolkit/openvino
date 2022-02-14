// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <utils/multidim_map.hpp>
#include <functional>
#include <tuple>
#include <array>

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
    bool needPrepareParams() const override;
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void initSupportedNV12Impls();
    void initSupportedI420Impls();

private:
    using ConverterBuilder = std::function<Converter*(MKLDNNNode *)>;
    using SupportedImpls = multidim_map<impl_desc_type,                             // Implementation type
                                        Algorithm,                                  // Algorithm: ColorConvertXXX
                                        InferenceEngine::Precision::ePrecision,     // Precision: FP32/U8
                                        bool,                                       // true - SinglePlaneConvert, false - TwoPlaneConvert/ThreePlaneConvert
                                        ConverterBuilder>;

    std::unique_ptr<Converter> _impl;
    SupportedImpls _supportedImpls;
};

class MKLDNNColorConvertNode::Converter {
public:
    using PrimitiveDescs = std::vector<std::tuple<std::vector<PortConfigurator>,    // Input port configurator
                                                  std::vector<PortConfigurator>,    // Output port configurator
                                                  impl_desc_type,                   // Implementation type
                                                  bool>>;                           // // true - SinglePlaneConvert, false - TwoPlaneConvert/ThreePlaneConvert
    using Shapes = std::vector<VectorDims>;

    static constexpr size_t N_DIM = 0;
    static constexpr size_t H_DIM = 1;
    static constexpr size_t W_DIM = 2;
    static constexpr size_t C_DIM = 3;

    using ColorFormat = std::array<uint8_t, 3>;

    Converter(MKLDNNNode *node, const ColorFormat & colorFormat);
    virtual ~Converter() = default;
    InferenceEngine::Precision inputPrecision(size_t idx) const;
    InferenceEngine::Precision outputPrecision(size_t idx) const;
    const void * input(size_t idx) const;
    void * output(size_t idx) const;
    const VectorDims & inputDims(size_t idx) const;
    virtual Shapes shapeInfer() const = 0;
    virtual void execute(mkldnn::stream strm) = 0;

protected:
    MKLDNNNode *_node;
    ColorFormat _colorFormat;   // RGB: {0,1,2}, BGR: {2,1,0}
};

}  // namespace MKLDNNPlugin
