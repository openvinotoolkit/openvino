// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <array>
#include <functional>
#include <tuple>
#include <utils/multidim_map.hpp>

namespace ov {
namespace intel_cpu {
namespace node {

class ColorConvert : public Node {
public:
    ColorConvert(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    class Converter;

public:
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    void initSupportedNV12Impls();
    void initSupportedI420Impls();

private:
    using ConverterBuilder = std::function<Converter*(Node*)>;
    using SupportedImpls = multidim_map<impl_desc_type,       // Implementation type
                                        Algorithm,            // Algorithm: ColorConvertXXX
                                        ov::element::Type_t,  // element type: f32/u8
                                        bool,  // true - SinglePlaneConvert, false - TwoPlaneConvert/ThreePlaneConvert
                                        ConverterBuilder>;

    std::unique_ptr<Converter> _impl;
    SupportedImpls _supportedImpls;
};

class ColorConvert::Converter {
public:
    using PrimitiveDescs =
        std::vector<std::tuple<std::vector<PortConfigurator>,  // Input port configurator
                               std::vector<PortConfigurator>,  // Output port configurator
                               impl_desc_type,                 // Implementation type
                               bool>>;  // // true - SinglePlaneConvert, false - TwoPlaneConvert/ThreePlaneConvert
    using Shapes = std::vector<VectorDims>;

    static constexpr size_t N_DIM = 0;
    static constexpr size_t H_DIM = 1;
    static constexpr size_t W_DIM = 2;
    static constexpr size_t C_DIM = 3;

    using ColorFormat = std::array<uint8_t, 3>;

    Converter(Node* node, const ColorFormat& colorFormat);
    virtual ~Converter() = default;
    ov::element::Type inputPrecision(size_t idx) const;
    ov::element::Type outputPrecision(size_t idx) const;
    const void* input(size_t idx) const;
    void* output(size_t idx) const;
    const VectorDims& inputDims(size_t idx) const;
    virtual void execute(const dnnl::stream& strm) = 0;

protected:
    Node* _node;
    ColorFormat _colorFormat;  // RGB: {0,1,2}, BGR: {2,1,0}
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
