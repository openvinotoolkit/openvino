// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_ngraph_utils.hpp"
#include <ie_cnn_layer_builder.h>
#include "blob_factory.hpp"
#include "ie_memcpy.h"

#include <sstream>
#include <limits>
#include <set>

#include <ngraph.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/op/avg_pool.hpp>
#include <ngraph/op/broadcast.hpp>
#include <ngraph/op/concat.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/divide.hpp>
#include <ngraph/op/dot.hpp>
#include <ngraph/op/exp.hpp>
#include <ngraph/op/experimental/dyn_reshape.hpp>
#include <ngraph/op/experimental/layers/detection_output.hpp>
#include <ngraph/op/experimental/layers/interpolate.hpp>
#include <ngraph/op/experimental/layers/prior_box.hpp>
#include <ngraph/op/experimental/layers/prior_box_clustered.hpp>
#include <ngraph/op/experimental/layers/proposal.hpp>
#include <ngraph/op/experimental/shape_of.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/fused/clamp.hpp>
#include <ngraph/op/fused/conv_fused.hpp>
#include <ngraph/op/fused/elu.hpp>
#include <ngraph/op/fused/group_conv.hpp>
#include <ngraph/op/fused/leaky_relu.hpp>
#include <ngraph/op/fused/mvn.hpp>
#include <ngraph/op/fused/prelu.hpp>
#include <ngraph/op/fused/split.hpp>
#include <ngraph/op/lrn.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/multiply.hpp>
#include <ngraph/op/pad.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/power.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/sigmoid.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/op/subtract.hpp>
#include <ngraph/op/tanh.hpp>

#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/group_conv_bias.hpp"
#include "ngraph_ops/matmul_bias.hpp"
#include "ngraph_ops/power.hpp"
#include "ngraph_ops/prior_box_clustered_ie.hpp"
#include "ngraph_ops/prior_box_ie.hpp"
#include "ngraph_ops/quantize_conv_bias_fused.hpp"
#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/tile_ie.hpp"
#include "ngraph_ops/interp.hpp"
#include "ngraph_ops/crop_ie.hpp"



std::map<std::string, std::string> InferenceEngine::Builder::convertParameters2Strings(const std::map<std::string, Parameter>& parameters) {
    std::map<std::string, std::string> oldParams;
    for (const auto& param : parameters) {
        // skip blobs and ports
        if (param.second.is<Blob::CPtr>() || param.second.is<Blob::Ptr>() || param.second.is<std::vector<Port>>()
                || param.second.is<PreProcessInfo>())
            continue;
        if (param.second.is<std::string>() || param.second.is<std::vector<std::string>>()) {
            oldParams[param.first] = Builder::convertParameter2String<std::string>(param.second);
        } else if (param.second.is<int>() || param.second.is<std::vector<int>>()) {
            oldParams[param.first] = Builder::convertParameter2String<int>(param.second);
        } else if (param.second.is<float>() || param.second.is<std::vector<float>>()) {
            oldParams[param.first] = Builder::convertParameter2String<float>(param.second);
        } else if (param.second.is<unsigned int>() || param.second.is<std::vector<unsigned int>>()) {
            oldParams[param.first] = Builder::convertParameter2String<unsigned int>(param.second);
        } else if (param.second.is<size_t>() || param.second.is<std::vector<size_t>>()) {
            oldParams[param.first] = Builder::convertParameter2String<size_t>(param.second);
        } else if (param.second.is<bool>() || param.second.is<std::vector<bool>>()) {
            oldParams[param.first] = Builder::convertParameter2String<bool>(param.second);
        } else {
            THROW_IE_EXCEPTION << "Parameter " << param.first << " has unsupported parameter type!";
        }
    }
    return oldParams;
}

InferenceEngine::Builder::Layer InferenceEngine::Builder::builderFromCNNLayer(const CNNLayerPtr& cnnLayer) {
    Builder::Layer layer(cnnLayer->type, cnnLayer->name);
    std::vector<Port> inputPorts;
    for (const auto& data : cnnLayer->insData) {
        auto lockedData = data.lock();
        if (!lockedData)
            continue;
        inputPorts.emplace_back(lockedData->getTensorDesc().getDims());
    }

    std::vector<Port> outputPorts;
    for (const auto& data : cnnLayer->outData) {
        outputPorts.emplace_back(data->getTensorDesc().getDims());
    }

    size_t inputsCount = inputPorts.size();
    std::map<std::string, Blob::Ptr> blobs = cnnLayer->blobs;
    if (blobs.find("weights") != blobs.end()) {
        auto port = Port();
        port.setParameter("type", "weights");
        inputPorts.push_back(port);
    }
    if (blobs.find("biases") != blobs.end()) {
        if (inputsCount == inputPorts.size()) {
            auto port = Port();
            port.setParameter("type", "weights");
            inputPorts.push_back(port);
        }

        auto port = Port();
        port.setParameter("type", "biases");
        inputPorts.push_back(port);
    }
    for (const auto& it : blobs) {
        if (it.first == "weights" || it.first == "biases")
            continue;
        auto port = Port();
        port.setParameter("type", it.first);
        inputPorts.emplace_back(port);
    }

    std::map<std::string, Parameter> params;
    for (const auto& it : cnnLayer->params) {
        params[it.first] = it.second;
    }

    layer.setInputPorts(inputPorts).setOutputPorts(outputPorts).setParameters(params);

    Builder::ConverterRegister::convert(cnnLayer, layer);

    return layer;
}

InferenceEngine::Builder::ConverterRegister::ConverterRegister(const std::string& type,
        const std::function<void(const CNNLayerPtr&, Layer&)>& converter) {
    if (getConvertersHolder().converters.find(type) == getConvertersHolder().converters.end())
        getConvertersHolder().converters[type] = converter;
}

InferenceEngine::Builder::ConvertersHolder &InferenceEngine::Builder::ConverterRegister::getConvertersHolder() {
    static Builder::ConvertersHolder holder;
    return holder;
}

namespace InferenceEngine {
namespace Builder {

template <>
inline std::string INodeConverter::asString<double>(const double& value) {
    std::ostringstream sStrm;
    sStrm.precision(std::numeric_limits<double>::digits10);
    sStrm << std::fixed << value;
    std::string result = sStrm.str();

    auto pos = result.find_last_not_of("0");
    if (pos != std::string::npos)
        result.erase(pos + 1);

    pos = result.find_last_not_of(".");
    if (pos != std::string::npos)
        result.erase(pos + 1);

    return result;
}

template <>
inline std::string INodeConverter::asString<float>(const float& value) {
    return asString(static_cast<double>(value));
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Parameter>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const InferenceEngine::Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Input", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Constant>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Const", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Constant>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto *data = castedLayer->get_data_ptr();
    auto dataPrecision  = details::ngraph::convertPrecision(castedLayer->get_element_type());
    SizeVector dataShape = castedLayer->get_shape();

    Blob::Ptr dataBlb = make_blob_with_precision(TensorDesc(dataPrecision, dataShape, TensorDesc::getLayoutByDims(dataShape)));
    dataBlb->allocate();
    ie_memcpy(dataBlb->buffer(), dataBlb->byteSize(), data, dataBlb->byteSize());
    res->blobs["custom"] = dataBlb;
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sigmoid>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Sigmoid", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Tanh>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "TanH", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::LeakyRelu>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "ReLU", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReLULayer>(params);

    NodeConverter<ngraph::op::Constant> converter;
    const auto orderNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(orderNode)) {
        const auto& orderLayer = converter.createLayer(orderNode, precision);
        auto order = orderLayer->blobs["custom"];
        res->params["negative_slope"] = asString(order->buffer().as<float *>()[0]);
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Relu>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "ReLU", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReLULayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Exp>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Exp", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::MVN>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "MVN", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::MVNLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::MVN>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["eps"] = asString(castedLayer->get_eps());
    res->params["across_channels"] = asString(castedLayer->get_across_channels());
    res->params["normalize_variance"] = asString(castedLayer->get_normalize_variance());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::LRN>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Norm", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::NormLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::LRN>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["alpha"] = asString(castedLayer->get_alpha());
    res->params["beta"] = asString(castedLayer->get_beta());
    res->params["local-size"] = asString(castedLayer->get_nsize());
    res->params["region"] = "across";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::CropIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                             const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Crop", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CropLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::CropIE>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->axes) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["axis"] = value;

    value.clear();
    for (const auto& val : castedLayer->dim) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["dim"] = value;

    value.clear();
    for (const auto& val : castedLayer->offset) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["offset"] = value;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Clamp>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Clamp", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ClampLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Clamp>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["min"] = asString(castedLayer->get_min());
    res->params["max"] = asString(castedLayer->get_max());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Softmax>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "SoftMax", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::SoftMaxLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Softmax>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;
    std::string value;
    for (const auto& val : castedLayer->get_axes()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["axis"] = value;
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Subtract>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "sub";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Power>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "pow";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Maximum>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "max";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Divide>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "div";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Multiply>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "prod";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Add>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "sum";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Broadcast>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Reshape", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReshapeLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Broadcast>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Squeeze>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                              const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Squeeze", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Squeeze>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Unsqueeze>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                              const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Unsqueeze", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Unsqueeze>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ConvolutionBackpropData>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                       const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Deconvolution", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::DeconvolutionLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropData>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_padding_below_forward()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_padding_above_forward()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_movement_strides_forward()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_dilation_strides_forward()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["dilations"] = value;

    // Restore kernel size and output
    const auto& shape = castedLayer->get_input_shape(0);
    res->params["output"] = asString(shape[1]);

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty())
            value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = castedLayer->get_inputs()[0].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode, precision);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GroupConvolution>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                       const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Convolution", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ConvolutionLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_padding_below()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_padding_above()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    switch (castedLayer->get_pad_type()) {
        case ngraph::op::PadType::SAME_UPPER:
            res->params["auto_pad"] = "same_upper";
            break;
        case ngraph::op::PadType::SAME_LOWER:
            res->params["auto_pad"] = "same_lower";
            break;
        default:
            break;
    }

    value.clear();
    for (const auto& val : castedLayer->get_window_movement_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_dilation_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["dilations"] = value;

    // Restore kernel size and output
    const auto& shape = castedLayer->get_input_shape(1);
    res->params["output"] = asString(shape[0]);
    res->params["group"] = asString(castedLayer->get_groups());

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty())
            value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode, precision);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Convolution>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                  const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Convolution", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ConvolutionLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Convolution>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_padding_below()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_padding_above()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    switch (castedLayer->get_pad_type()) {
        case ngraph::op::PadType::SAME_UPPER:
            res->params["auto_pad"] = "same_upper";
            break;
        case ngraph::op::PadType::SAME_LOWER:
            res->params["auto_pad"] = "same_lower";
            break;
        default:
            break;
    }

    value.clear();
    for (const auto& val : castedLayer->get_window_movement_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_dilation_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["dilations"] = value;

    // Restore kernel size and output
    const auto& shape = castedLayer->get_input_shape(1);
    res->params["output"] = asString(shape[0]);

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty())
            value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode, precision);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }
    return res;
}


template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GroupConvolutionBias>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                           const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Convolution", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ConvolutionLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_padding_below()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_padding_above()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_movement_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_dilation_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["dilations"] = value;

    // Restore kernel size and output
    const auto& shape = castedLayer->get_input_shape(1);
    res->params["output"] = asString(shape[0]);
    res->params["group"] = asString(castedLayer->get_groups());

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty())
            value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode, precision);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }

    const auto biasNode = castedLayer->get_inputs()[2].get_output().get_node();
    if (converter.canCreate(biasNode)) {
        const auto& bias = converter.createLayer(biasNode, precision);
        res->blobs["biases"] = bias->blobs["custom"];
        res->_biases = bias->blobs["custom"];
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ConvolutionBias>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                      const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Convolution", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ConvolutionLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::ConvolutionBias>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_padding_below()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_padding_above()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_movement_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_dilation_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["dilations"] = value;

    // Restore kernel size and output
    const auto& shape = castedLayer->get_input_shape(1);
    res->params["output"] = asString(shape[0]);

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty())
            value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode, precision);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }

    const auto biasNode = castedLayer->get_inputs()[2].get_output().get_node();
    if (converter.canCreate(biasNode)) {
        const auto& bias = converter.createLayer(biasNode, precision);
        res->blobs["biases"] = bias->blobs["custom"];
        res->_biases = bias->blobs["custom"];
    }

    return res;
}


template <>
CNNLayer::Ptr NodeConverter<ngraph::op::AvgPool>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Pooling", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PoolingLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::AvgPool>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_padding_below()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_padding_above()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_movement_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_shape()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["kernel"] = value;

    auto includePad = castedLayer->get_include_padding_in_avg_computation();
    res->params["exclude-pad"] = includePad ? "false" : "true";
    res->params["pool-method"] = "avg";
    res->params["rounding_type"] = castedLayer->get_ceil_mode() ? "ceil" : "floor";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::MaxPool>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Pooling", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PoolingLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::MaxPool>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_padding_below()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_padding_above()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_movement_strides()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_window_shape()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["kernel"] = value;
    res->params["pool-method"] = "max";
    res->params["rounding_type"] = castedLayer->get_ceil_mode() ? "ceil" : "floor";

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PRelu>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "PReLU", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PReLULayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::PRelu>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode, precision);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Split>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Split", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::SplitLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Split>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->get_axis());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Concat>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Concat", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ConcatLayer>(params);

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Concat>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->get_concatenation_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Reshape>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Reshape", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReshapeLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ShapeOf>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "ShapeOf", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::DynReshape>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Reshape", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReshapeLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Pad>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Pad", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PadLayer>(params);

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Pad>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    switch (castedLayer->get_pad_mode()) {
    case ngraph::op::PadMode::EDGE:
        res->params["pad_mode"] = "edge";
        break;
    case ngraph::op::PadMode::REFLECT:
        res->params["pad_mode"] = "reflect";
        break;
    case ngraph::op::PadMode::CONSTANT:
        res->params["pad_mode"] = "constant";
        break;
    }
    std::string pad;
    for (const auto& p : castedLayer->get_padding_below()) {
        if (!pad.empty())
            pad += ",";
        pad += asString(p);
    }
    res->params["pads_begin"] = pad;

    pad.clear();
    for (const auto& p : castedLayer->get_padding_above()) {
        if (!pad.empty())
            pad += ",";
        pad += asString(p);
    }
    res->params["pads_end"] = pad;

    NodeConverter<ngraph::op::Constant> converter;
    const auto defValNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(defValNode)) {
        const auto& weightsLayer = converter.createLayer(defValNode, precision);
        const auto& weights = weightsLayer->blobs["custom"];
        switch (weights->getTensorDesc().getPrecision()) {
        case Precision::U8:
            res->params["pad_value"] = asString(weights->buffer().as<uint8_t *>()[0]);
            break;
        case Precision::U16:
            res->params["pad_value"] = asString(weights->buffer().as<uint16_t *>()[0]);
            break;
        case Precision::I8:
            res->params["pad_value"] = asString(weights->buffer().as<int8_t *>()[0]);
            break;
        case Precision::I16:
            res->params["pad_value"] = asString(weights->buffer().as<int16_t *>()[0]);
            break;
        case Precision::I32:
            res->params["pad_value"] = asString(weights->buffer().as<int32_t *>()[0]);
            break;
        case Precision::I64:
            res->params["pad_value"] = asString(weights->buffer().as<int64_t *>()[0]);
            break;
        case Precision::FP32:
            res->params["pad_value"] = asString(weights->buffer().as<float *>()[0]);
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported precision!";
        }
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ScaleShiftIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                             const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "ScaleShift", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ScaleShiftLayer>(params);

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weightsLayer = converter.createLayer(weightsNode, precision);
        res->blobs["weights"] = weightsLayer->blobs["custom"];
        res->_weights = weightsLayer->blobs["custom"];
    }

     const auto biasNode = layer->get_inputs()[2].get_output().get_node();
     if (converter.canCreate(biasNode)) {
         const auto& bias = converter.createLayer(biasNode, precision);
         res->blobs["biases"] = bias->blobs["custom"];
         res->_biases = bias->blobs["custom"];
     }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Elu>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                             const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "ELU", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    NodeConverter<ngraph::op::Constant> converter;
    const auto orderNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(orderNode)) {
        const auto& orderLayer = converter.createLayer(orderNode, precision);
        auto order = orderLayer->blobs["custom"];
        res->params["alpha"] = asString(order->buffer().as<float *>()[0]);
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::DetectionOutput>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                             const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "DetectionOutput", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::DetectionOutput>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attr = castedLayer->get_attrs();
    std::string param;

    res->params["num_classes"] = asString(attr.num_classes);
    res->params["background_label_id"] = asString(attr.background_label_id);
    res->params["top_k"] = asString(attr.top_k);
    res->params["variance_encoded_in_target"] = (attr.variance_encoded_in_target ? "1" : "0");
    for (const auto& val : attr.keep_top_k) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["keep_top_k"] = param;
    res->params["code_type"] = attr.code_type;
    res->params["share_location"] = (attr.share_location ? "1" : "0");
    res->params["nms_threshold"] = asString(attr.nms_threshold);
    res->params["confidence_threshold"] = asString(attr.confidence_threshold);
    res->params["clip_after_nms"] = (attr.clip_after_nms ? "1" : "0");
    res->params["clip_before_nms"] = (attr.clip_before_nms ? "1" : "0");
    res->params["decrease_label_id"] = (attr.decrease_label_id ? "1" : "0");
    res->params["normalized"] = (attr.normalized ? "1" : "0");
    res->params["input_height"] = asString(attr.input_height);
    res->params["input_width"] = asString(attr.input_width);
    res->params["objectness_score"] = asString(attr.objectness_score);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Transpose>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                             const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Permute", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    NodeConverter<ngraph::op::Constant> converter;
    const auto orderNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(orderNode)) {
        const auto& orderLayer = converter.createLayer(orderNode, precision);
        auto order = orderLayer->blobs["custom"];
        int64_t* data = order->buffer().as<int64_t *>();
        std::string orderStr;
        for (size_t i = 0; i < order->size(); i++) {
            if (!orderStr.empty())
                orderStr += ",";
            orderStr += asString(data[i]);
        }
        res->params["order"] = orderStr;
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Proposal>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                 const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Proposal", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Proposal>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attr = castedLayer->get_attrs();
    std::string param;
    for (const auto& val : attr.ratio) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["ratio"] = param;

    param.clear();
    for (const auto& val : attr.scale) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["scale"] = param;

    res->params["base_size"] = asString(attr.base_size);
    res->params["pre_nms_topn"] = asString(attr.pre_nms_topn);
    res->params["post_nms_topn"] = asString(attr.post_nms_topn);
    res->params["nms_thresh"] = asString(attr.nms_thresh);
    res->params["feat_stride"] = asString(attr.feat_stride);
    res->params["min_size"] = asString(attr.min_size);
    res->params["box_size_scale"] = asString(attr.box_size_scale);
    res->params["box_coordinate_scale"] = asString(attr.box_coordinate_scale);
    res->params["clip_before_nms"] = asString(attr.clip_before_nms ? 1 : 0);
    res->params["clip_after_nms"] = asString(attr.clip_after_nms ? 1 : 0);
    res->params["normalize"] = asString(attr.normalize ? 1 : 0);
    res->params["framework"] = attr.framework;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBoxClusteredIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                 const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "PriorBoxClustered", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::PriorBoxClusteredIE>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attr = castedLayer->get_attrs();
    std::string param;
    for (const auto& val : attr.widths) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["width"] = param;

    param.clear();
    for (const auto& val : attr.heights) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["height"] = param;

    param.clear();
    for (const auto& val : attr.variances) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["variance"] = param;

    res->params["step_w"] = asString(attr.step_widths);
    res->params["step_h"] = asString(attr.step_heights);
    res->params["offset"] = asString(attr.offset);
    res->params["clip"] = asString(attr.clip ? 1 : 0);
    res->params["flip"] = "0";

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBoxClustered>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                 const Precision& precision) const {
    THROW_IE_EXCEPTION << "PriorBoxClustered operation must be converted to PriorBoxClusteredIE operation.";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBoxIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                 const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "PriorBox", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::PriorBoxIE>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attr = castedLayer->get_attrs();
    std::string param;
    for (const auto& val : attr.max_size) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["max_size"] = param;

    param.clear();
    for (const auto& val : attr.min_size) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["min_size"] = param;

    param.clear();
    for (const auto& val : attr.aspect_ratio) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["aspect_ratio"] = param;

    param.clear();
    for (const auto& val : attr.variance) {
        if (!param.empty())
            param += ",";
        param += asString(val);
    }
    res->params["variance"] = param;

    res->params["step"] = asString(attr.step);
    res->params["offset"] = asString(attr.offset);
    res->params["clip"] = asString(attr.clip ? 1 : 0);
    res->params["flip"] = asString(attr.flip ? 1 : 0);
    res->params["scale_all_sizes"] = asString(attr.scale_all_sizes ? 1 : 0);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBox>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                 const Precision& precision) const {
    THROW_IE_EXCEPTION << "PriorBox operation must be converted to PriorBoxIE operation.";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PowerIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                              const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Power", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PowerLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::PowerIE>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["power"] = asString(castedLayer->power);
    res->params["scale"] = asString(castedLayer->scale);
    res->params["shift"] = asString(castedLayer->shift);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Eltwise>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                              const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Eltwise>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string type;
    switch (castedLayer->eltwise_type) {
        case ELTWISE_TYPE::Sum:
            type = "sum";
            break;
        case ELTWISE_TYPE::Prod:
            type = "prod";
            break;
        default:
            THROW_IE_EXCEPTION << "Not supported eltwise type!";
    }

    res->params["operation"] = type;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::TileIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                             const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Tile", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TileLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::TileIE>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->axis);
    res->params["tiles"] = asString(castedLayer->tiles);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Interp>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                             const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "Resample", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Interp>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attrs = castedLayer->get_attrs();
    if (std::set<std::string>{"nearest", "cubic", "area"}.count(attrs.mode)) {
        auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
        if (attrs.pad_beg != 0) {
            THROW_IE_EXCEPTION << "Resample do not supports pad";
        }

        if (attrs.pad_end != 0) {
            THROW_IE_EXCEPTION << "Resample do not supports pad";
        }

        if (attrs.align_corners) {
            THROW_IE_EXCEPTION << "Resample do not supports align corners";
        }

        if (attrs.mode == "nearest") {
            res->params["type"] = "caffe.ResampleParameter.NEAREST";
        } else if (attrs.mode == "cubic") {
            res->params["type"] = "caffe.ResampleParameter.CUBIC";
        } else if (attrs.mode == "area") {
            res->params["type"] = "caffe.ResampleParameter.AREA";
        } else if (attrs.mode == "linear") {
            res->params["type"] = "caffe.ResampleParameter.LINEAR";
        }

        res->params["height"] = asString(attrs.height);
        res->params["width"] = asString(attrs.width);

        res->params["antialias"] = attrs.antialias ? "1" : "0";
        return res;
    }

    if (attrs.antialias) {
        THROW_IE_EXCEPTION << "Interp do not support antialias";
    }

    params = {layer->get_friendly_name(), "Interp", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    res->params["height"] = asString(attrs.height);
    res->params["width"] = asString(attrs.width);
    res->params["pad_beg"] = asString(attrs.pad_beg);
    res->params["pad_end"] = asString(attrs.pad_end);
    res->params["align_corners"] = attrs.align_corners ? "1" : "0";

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Interpolate>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                                                 const Precision& precision) const {
    THROW_IE_EXCEPTION << "Interpolate operation should be converted to Interp";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Dot>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
            const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "FullyConnected", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::FullyConnectedLayer>(params);
    res->params["alpha"] = "0";
    res->params["beta"] = "0";

    if (layer->get_input_size() < 2)
        THROW_IE_EXCEPTION << "Cannot convert an incorrect Dot layer!";

    res->params["out-size"] = asString(layer->get_inputs()[1].get_output().get_shape()[1]);
    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weightsLayer = converter.createLayer(weightsNode, precision);
        const auto& weights = weightsLayer->blobs["custom"];
        if (weights->getTensorDesc().getDims().size() != 2)
            THROW_IE_EXCEPTION << "Unsuppoted weights for FullyConnected layers!";

        const auto dims = weights->getTensorDesc().getDims();
        const auto* origData = weights->buffer().as<uint8_t *>();

        Blob::Ptr dataBlb = make_blob_with_precision(TensorDesc(weights->getTensorDesc().getPrecision(),
                                                                {dims[1], dims[0]},
                                                                weights->getTensorDesc().getLayout()));
        dataBlb->allocate();

        auto* transpData = dataBlb->buffer().as<uint8_t *>();
        size_t transpDataSize = dataBlb->byteSize();
        size_t elementSize = dataBlb->element_size();
        for (size_t j = 0, k = 0; j < dims[1]; j++) {
            for (size_t i = 0; i < dims[0]; i++, k++) {
                const size_t offset_src = (i*dims[1] + j) * elementSize;
                const size_t offset_dst = k * elementSize;
                if (transpDataSize < offset_dst)  {
                    // zero out dest if error detected
                    memset(transpData, 0, transpDataSize);
                    THROW_IE_EXCEPTION << "Size error";
                }
                ie_memcpy(&transpData[offset_dst], transpDataSize - offset_dst,
                    &origData[offset_src], elementSize);
            }
        }

        res->blobs["weights"] = dataBlb;
        res->_weights = dataBlb;
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::MatmulBias>::createLayer(const std::shared_ptr<ngraph::Node>& layer,
                                             const Precision& precision) const {
    LayerParams params = {layer->get_friendly_name(), "FullyConnected", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::FullyConnectedLayer>(params);
    res->params["alpha"] = "0";
    res->params["beta"] = "0";

    if (layer->get_input_size() < 2)
        THROW_IE_EXCEPTION << "Cannot convert an incorrect Dot layer!";

    res->params["out-size"] = asString(layer->get_inputs()[1].get_output().get_shape()[1]);
    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weightsLayer = converter.createLayer(weightsNode, precision);
        const auto& weights = weightsLayer->blobs["custom"];
        if (weights->getTensorDesc().getDims().size() != 2)
            THROW_IE_EXCEPTION << "Unsuppoted weights for FullyConnected layers!";

        const auto dims = weights->getTensorDesc().getDims();
        const auto* origData = weights->buffer().as<uint8_t *>();

        Blob::Ptr dataBlb = make_blob_with_precision(TensorDesc(weights->getTensorDesc().getPrecision(),
                                                                {dims[1], dims[0]},
                                                                weights->getTensorDesc().getLayout()));
        dataBlb->allocate();

        auto* transpData = dataBlb->buffer().as<uint8_t *>();
        size_t transpDataSize = dataBlb->byteSize();
        size_t elementSize = dataBlb->element_size();
        for (size_t j = 0, k = 0; j < dims[1]; j++) {
            for (size_t i = 0; i < dims[0]; i++, k++) {
                const size_t offset_src = (i*dims[1] + j) * elementSize;
                const size_t offset_dst = k * elementSize;
                if (transpDataSize < offset_dst)  {
                    // zero out dest if error detected
                    memset(transpData, 0, transpDataSize);
                    THROW_IE_EXCEPTION << "Size error";
                }
                ie_memcpy(&transpData[offset_dst], transpDataSize - offset_dst,
                    &origData[offset_src], elementSize);
            }
        }

        res->blobs["weights"] = dataBlb;
        res->_weights = dataBlb;
    }
    if (layer->get_input_size() == 3) {
        const auto biasNode = layer->get_inputs()[2].get_output().get_node();
        if (converter.canCreate(biasNode)) {
            const auto &bias = converter.createLayer(biasNode, precision);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
    }
    return res;
}

}  // namespace Builder
}  // namespace InferenceEngine
