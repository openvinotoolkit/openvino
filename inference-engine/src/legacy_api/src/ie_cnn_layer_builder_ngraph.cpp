// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <cmath>
#include <set>
#include <sstream>
#include <utility>

#include "legacy/ngraph_ops/fully_connected.hpp"
#include "legacy/ngraph_ops/interp.hpp"
#include <transformations/rt_info/primitives_priority_attribute.hpp>
#include "legacy/ngraph_ops/scaleshift.hpp"

#include "exec_graph_info.hpp"

#include <cnn_network_ngraph_impl.hpp>
#include <precision_utils.h>
#include <cpp/ie_cnn_network.h>
#include <ngraph/ngraph.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset5.hpp>

#include <legacy/convert_function_to_cnn_network.hpp>
#include "legacy/graph_transformer.h"
#include "legacy/graph_tools.hpp"
#include "legacy/net_pass.h"
#include <legacy/cnn_network_impl.hpp>
#include <ie_cnn_layer_builder_ngraph.h>

namespace InferenceEngine {
namespace Builder {

template <>
std::string asString<double>(const double& value) {
    std::ostringstream sStrm;
    sStrm.precision(std::numeric_limits<double>::digits10);
    sStrm << std::fixed << value;
    std::string result = sStrm.str();

    auto pos = result.find_last_not_of("0");
    if (pos != std::string::npos) result.erase(pos + 1);

    pos = result.find_last_not_of(".");
    if (pos != std::string::npos) result.erase(pos + 1);

    return result;
}

template <>
std::string asString<float>(const float& value) {
    return asString(static_cast<double>(value));
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ScaleShiftIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ScaleShift",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ScaleShiftLayer>(params);

    const auto weightsNode = layer->input_value(1).get_node_shared_ptr();
    InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);
    const auto biasNode = layer->input_value(2).get_node_shared_ptr();
    InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ShuffleChannels>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ShuffleChannels", details::convertPrecision(layer->get_output_element_type(0))};

    auto res = std::make_shared<InferenceEngine::ShuffleChannelsLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ShuffleChannels>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = std::to_string(castedLayer->get_axis());
    res->params["group"] = std::to_string(castedLayer->get_group());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ResampleV2>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Resample", details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ResampleV2>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attrs = castedLayer->get_attrs();

    res->params["antialias"] = attrs.antialias ? "1" : "0";
    if (attrs.mode == "nearest") {
        res->params["type"] = "caffe.ResampleParameter.NEAREST";
    } else if (attrs.mode == "cubic") {
        res->params["type"] = "caffe.ResampleParameter.CUBIC";
    } else if (attrs.mode == "area") {
        res->params["type"] = "caffe.ResampleParameter.AREA";
    } else if (attrs.mode == "linear") {
        res->params["type"] = "caffe.ResampleParameter.LINEAR";
    }

    res->params["factor"] = asString(attrs.factor);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v4::Interpolate>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Interpolate",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v4::Interpolate>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attrs = castedLayer->get_attrs();

    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    switch (attrs.mode) {
        case ::ngraph::op::v4::Interpolate::InterpolateMode::nearest: {
            res->params["mode"] = "nearest";
            break;
        }
        case ::ngraph::op::v4::Interpolate::InterpolateMode::linear: {
            res->params["mode"] = "linear";
            break;
        }
        case ::ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx: {
            res->params["mode"] = "linear_onnx";
            break;
        }
        case ::ngraph::op::v4::Interpolate::InterpolateMode::cubic: {
            res->params["mode"] = "cubic";
            break;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported mode for Interpolate op";
            break;
    }

    switch (attrs.shape_calculation_mode) {
        case ::ngraph::op::v4::Interpolate::ShapeCalcMode::sizes: {
            res->params["shape_calculation_mode"] = "sizes";
            break;
        }
        case ::ngraph::op::v4::Interpolate::ShapeCalcMode::scales: {
            res->params["shape_calculation_mode"] = "scales";
            break;
        }
        default:
            THROW_IE_EXCEPTION << "Unsupported shape_calculation_mode for Interpolate op";
            break;
    }

    switch (attrs.coordinate_transformation_mode) {
        case ::ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel: {
            res->params["coordinate_transformation_mode"] = "half_pixel";
            break;
        }
        case ::ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel: {
            res->params["coordinate_transformation_mode"] = "pytorch_half_pixel";
            break;
        }
        case ::ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric: {
            res->params["coordinate_transformation_mode"] = "asymmetric";
            break;
        }
        case ::ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn: {
            res->params["coordinate_transformation_mode"] = "tf_half_pixel_for_nn";
            break;
        }
        case ::ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners: {
            res->params["coordinate_transformation_mode"] = "align_corners";
            break;
        }
        default:
            res->params["coordinate_transformation_mode"] = "half_pixel";
            break;
    }

    switch (attrs.nearest_mode) {
        case ::ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor: {
            res->params["nearest_mode"] = "round_prefer_floor";
            break;
        }
        case ::ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil: {
            res->params["nearest_mode"] = "round_prefer_ceil";
            break;
        }
        case ::ngraph::op::v4::Interpolate::NearestMode::floor: {
            res->params["nearest_mode"] = "floor";
            break;
        }
        case ::ngraph::op::v4::Interpolate::NearestMode::ceil: {
            res->params["nearest_mode"] = "ceil";
            break;
        }
        case ::ngraph::op::v4::Interpolate::NearestMode::simple: {
            res->params["nearest_mode"] = "simple";
            break;
        }
        default:
            res->params["nearest_mode"] = "round_prefer_floor";
            break;
    }

    res->params["antialias"] = attrs.antialias ? "True" : "False";

    std::string value;
    for (const auto& val : attrs.pads_begin) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : attrs.pads_end) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    res->params["cube_coeff"] = asString(attrs.cube_coeff);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ExecGraphInfoSerialization::ExecutionNode>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    auto castedLayer = ngraph::as_type_ptr<ExecGraphInfoSerialization::ExecutionNode>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert " << layer->get_friendly_name() << " layer ";

    auto & rtInfo = castedLayer->get_rt_info();
    if (rtInfo.count(ExecGraphInfoSerialization::LAYER_TYPE) == 0) {
        THROW_IE_EXCEPTION << "No " << ExecGraphInfoSerialization::LAYER_TYPE
            << " attribute is set in " << layer->get_friendly_name() << " node";
    }

    auto getStringValue = [] (const std::shared_ptr<ngraph::Variant> & variant) {
        auto castedVariant = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(variant);
        IE_ASSERT(castedVariant != nullptr);
        return castedVariant->get();
    };

    LayerParams params = { layer->get_friendly_name(),
                           getStringValue(rtInfo[ExecGraphInfoSerialization::LAYER_TYPE]),
                           details::convertPrecision(layer->get_output_element_type(0)) };
    rtInfo.erase(ExecGraphInfoSerialization::LAYER_TYPE);

    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    for (const auto & kvp : rtInfo) {
        auto castedVariant = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(kvp.second);
        // skip RT info which holds fusedNames, etc
        if (castedVariant)
            res->params[kvp.first] = getStringValue(castedVariant);
    }

    return res;
}
}  // namespace Builder
}  // namespace InferenceEngine
