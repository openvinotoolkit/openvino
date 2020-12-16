// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <cmath>
#include <set>
#include <sstream>
#include <utility>

#include "legacy/ngraph_ops/crop_ie.hpp"
#include "legacy/ngraph_ops/eltwise.hpp"
#include "legacy/ngraph_ops/fully_connected.hpp"
#include "legacy/ngraph_ops/gather_ie.hpp"
#include "legacy/ngraph_ops/gru_cell_ie.hpp"
#include "legacy/ngraph_ops/interp.hpp"
#include "legacy/ngraph_ops/lstm_cell_ie.hpp"
#include <transformations/rt_info/primitives_priority_attribute.hpp>
#include "legacy/ngraph_ops/normalize_ie.hpp"
#include "legacy/ngraph_ops/nms_ie.hpp"
#include "legacy/ngraph_ops/power.hpp"
#include "legacy/ngraph_ops/prior_box_clustered_ie.hpp"
#include "legacy/ngraph_ops/prior_box_ie.hpp"
#include "legacy/ngraph_ops/proposal_ie.hpp"
#include "legacy/ngraph_ops/relu_ie.hpp"
#include "legacy/ngraph_ops/selu_ie.hpp"
#include "legacy/ngraph_ops/scaleshift.hpp"
#include "legacy/ngraph_ops/tile_ie.hpp"
#include "legacy/ngraph_ops/rnn_cell_ie.hpp"

#include "generic_ie.hpp"
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
CNNLayer::Ptr NodeConverter<ngraph::op::Abs>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Abs",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GenericIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::GenericIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get layer " << layer->get_friendly_name();

    LayerParams params = {layer->get_friendly_name(), castedLayer->getType(),
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    if (castedLayer->getType() == "RNNCell")
        res = std::make_shared<InferenceEngine::RNNCell>(params);
    if (castedLayer->getType() == "GRUCell")
        res = std::make_shared<InferenceEngine::GRUCell>(params);

    auto weightableLayer = std::dynamic_pointer_cast<InferenceEngine::WeightableLayer>(res);

    for (const auto& param : castedLayer->getParameters()) {
        if (param.second.is<Blob::Ptr>()) {
            res->blobs[param.first] = param.second.as<Blob::Ptr>();
        } else if (param.second.is<Blob::CPtr>()) {
            res->blobs[param.first] = std::const_pointer_cast<Blob>(param.second.as<Blob::CPtr>());
        } else if (param.second.is<std::string>()) {
            res->params[param.first] = param.second.as<std::string>();
        }
        if (weightableLayer && param.first == "weights")
            weightableLayer->_weights = res->blobs[param.first];
        if (weightableLayer && param.first == "biases")
            weightableLayer->_biases = res->blobs[param.first];
    }
    return res;
}

CNNLayer::Ptr createSubGraphLayer(const std::shared_ptr<ngraph::Node>& layer) {
    auto find_input_idx = [](const CNNLayerPtr& where, const DataPtr& what) {
        auto it = std::find_if(where->insData.begin(), where->insData.end(), [&](const DataWeakPtr& wk_ptr) {
            auto layer_data = wk_ptr.lock();
            IE_ASSERT(layer_data != nullptr);
            return what->getName() == layer_data->getName();
        });
        if (it == where->insData.end()) {
            THROW_IE_EXCEPTION << "Input layer not found.";
        }

        return it - where->insData.begin();
    };

    auto tensor_iterator = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(layer);
    if (!tensor_iterator) {
        THROW_IE_EXCEPTION << "Cannot cast layer to TensorIterator.";
    }

    // inputs/outputs of TensorIterator (ngraph representation)
    auto parameters = tensor_iterator->get_function()->get_parameters();
    auto results = tensor_iterator->get_function()->get_results();

    // Convert body (ngraph representation) to CNNNetwork.
    // This network will contain nodes of type = "Input" and data nodes with wrong names.
    // IE TensorIterator doesn't include such nodes so we create CNNNetwork in a separate scope
    // to call the destructor and delete these "Input"/data nodes.

    TensorIterator::Body body;
    {
        CNNNetwork body_net(tensor_iterator->get_function());
        CNNNetwork net(InferenceEngine::details::convertFunctionToICNNNetwork(body_net.getFunction(), body_net));
        // Paranoid check for cycles
        bool res = CNNNetForestDFS(
            CNNNetGetAllInputLayers(net), [](const CNNLayerPtr& layer) {}, false);
        if (!res) {
            THROW_IE_EXCEPTION << "Loop detected. TensorIterator body should not contain loops.";
        }

        // Get inputs/outputs of cnn network
        auto in_info_map_with_parameters = net.getInputsInfo();
        auto out_info_map = net.getOutputsInfo();

        IE_ASSERT(in_info_map_with_parameters.size() == parameters.size());
        IE_ASSERT(out_info_map.size() == results.size());

        InferenceEngine::TensorIterator::Body temp_body;
        temp_body.inputs.resize(in_info_map_with_parameters.size());
        temp_body.outputs.resize(out_info_map.size());

        // Fill inputs/outs in order aligned with ng representation
        uint64_t counter = 0;
        for (const auto& param : parameters) {
            auto info = in_info_map_with_parameters.at(param->get_friendly_name());
            temp_body.inputs[counter++] = info->getInputData();
        }

        auto map_ng_result_to_ie_name = [] (std::shared_ptr<ngraph::op::v0::Result> res_op) {
            auto result = res_op->input(0).get_source_output();

            std::string name = result.get_node()->get_friendly_name();
            if (result.get_node()->get_output_size() > 1) {
                name += "." + std::to_string(result.get_index());
            }
            return name;
        };

        counter = 0;
        for (const auto& result : results) {
            auto data = out_info_map.at(map_ng_result_to_ie_name(result));
            temp_body.outputs[counter++] = data;
        }

        // This deep copy will hold all unreachable constants. See the comment in CopyTIBody function.
        body = InferenceEngine::NetPass::CopyTIBody(temp_body);

        // Check if data is really const layer holder
        auto is_constant_holder = [] (const DataPtr data) {
            return data->getPrecision() == Precision::UNSPECIFIED;
        };

        // Strip unreached node holder from Inputs node.
        auto holder = body.inputs.back();
        if (is_constant_holder(holder)) {
            auto& holder_map = getInputTo(holder);
            // remove_if
            for( auto it = holder_map.begin(); it != holder_map.end(); ) {
                if( it->second->type == "Input")
                    it = holder_map.erase(it);
                else
                    ++it;
            }
        }

        // TODO: Disable this WA after total switch onto Ngraph
        //   WA: Some plugins (like GPU) require matching of Data object name and producer Layer name.
        //       Data name is expected in format "[layer_name]" or "[layer_name].[port_idx]" in case
        //       of multiple inputs. We have to restore it if possible and ignore original names of
        //       Ngraph parameter and result ops.
        //       Will not change data name if:
        //        - data has several consumer layers
        //        - data has no consumer (example if data is straight used as output)
        //
        for (auto &in : body.inputs) {
            if (is_constant_holder(in))
                continue;

            const auto input_to = getInputTo(in);
            if (input_to.size() != 1)
                continue;

            const auto consumer_layer = input_to.begin()->second;
            const auto consumer_in_port_set = consumer_layer->insData;
            const auto found = std::find_if(consumer_in_port_set.begin(), consumer_in_port_set.end(),
                                      [&in] (const DataWeakPtr &wptr) { return wptr.lock() == in; });
            IE_ASSERT(found != consumer_in_port_set.end());
            const auto consumer_port_idx = std::distance(consumer_in_port_set.begin(), found);

            auto new_name = consumer_layer->name;
            if (consumer_in_port_set.size() > 1) {
                new_name += '.' + std::to_string(consumer_port_idx);
            }
            in->setName(new_name);
        }

        // TODO: this WA restore original precisions of outputs.
        //       convertFunctionToICNNNetwork has internal fallback policy for unsupported
        //       precisions for inputs/outputs ports. Particular for U8 will be translated
        //       to FP32. However Loop body has strong requirements for continue_condition
        //       port, it should be BOOL(U8).
        //
        for (int i = 0; i < results.size(); i++) {
            auto result = results[i];
            auto output = body.outputs[i];
            if (result->get_element_type() == ngraph::element::u8) {
                output->setPrecision(InferenceEngine::Precision::U8);
            }
        }
    }

    // Create Inference Engine representation of TensorIterator
    LayerParams params = {layer->get_friendly_name(), "TensorIterator",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TensorIterator>(params);
    res->body = body;

    // Port map: outputs
    for (const auto& desc : tensor_iterator->get_output_descriptions()) {
        auto body_output_idx = desc->m_body_value_index;

        std::string type_name = desc->get_type_info().name;
        if (type_name == "ConcatOutputDescription") {
            auto output_desc = ::ngraph::as_type_ptr<ngraph::op::TensorIterator::ConcatOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            res->output_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx),
                static_cast<int>(output_desc->m_axis), static_cast<int>(output_desc->m_stride),
                static_cast<int>(output_desc->m_start), static_cast<int>(output_desc->m_end),
                static_cast<int>(output_desc->m_part_size)});

        } else if (type_name == "BodyOutputDescription") {
            auto output_desc = ::ngraph::as_type_ptr<ngraph::op::TensorIterator::BodyOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            res->output_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx), -1, 1, 0, -1, 1});
        } else {
            THROW_IE_EXCEPTION << "Incorrect type of the output description.";
        }
    }

    // Port map : inputs and back edges
    for (const auto& desc : tensor_iterator->get_input_descriptions()) {
        auto body_input_index = desc->m_body_parameter_index;

        if (const auto slice_desc = std::dynamic_pointer_cast<ngraph::op::TensorIterator::SliceInputDescription>(desc)) {
            res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(slice_desc->m_input_index), static_cast<int>(body_input_index),
                static_cast<int>(slice_desc->m_axis), static_cast<int>(slice_desc->m_stride),
                static_cast<int>(slice_desc->m_start), static_cast<int>(slice_desc->m_end),
                static_cast<int>(slice_desc->m_part_size)});
        } else if (const auto merge_desc = std::dynamic_pointer_cast<ngraph::op::TensorIterator::MergedInputDescription>(desc)) {
            res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(merge_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});

            auto body_output_idx = merge_desc->m_body_value_index;

            res->back_edges.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(body_output_idx), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else if (const auto inv_desc = std::dynamic_pointer_cast<ngraph::op::TensorIterator::InvariantInputDescription>(desc)) {
            res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                    static_cast<int>(inv_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else {
            THROW_IE_EXCEPTION << "Incorrect type of the input description.";
        }
    }

    if (const auto loop_op = std::dynamic_pointer_cast<const ngraph::opset5::Loop>(layer)) {
        auto spec_port = loop_op->get_special_body_ports();
        if (spec_port.current_iteration_input_idx != -1) {
            auto ie_port_idx = spec_port.current_iteration_input_idx;
            res->params["loop_body_current_iteration_idx"] = std::to_string(ie_port_idx);
        }
        if (spec_port.body_condition_output_idx != -1) {
            auto body_output_idx = spec_port.body_condition_output_idx;
            res->params["loop_body_condition_output_idx"] = std::to_string(body_output_idx);
        }
        res->params["loop_trip_count_idx"] = "0";
        res->params["loop_execution_condition_idx"] = "1";
    }

    return res;
}

template<>
CNNLayer::Ptr NodeConverter<ngraph::op::TensorIterator>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    auto res = createSubGraphLayer(layer);
    res->type = "TensorIterator";
    return res;
}

template<>
CNNLayer::Ptr NodeConverter<ngraph::opset5::Loop>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    auto res = createSubGraphLayer(layer);
    res->type = "Loop";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Ceiling>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Ceiling",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Floor>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Floor",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sigmoid>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sigmoid",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Tanh>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "TanH",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ReLUIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReLU",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReLULayer>(params);

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ReLUIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["negative_slope"] = asString(castedLayer->get_slope());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Range>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Range",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Exp>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Exp",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::CropIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Crop",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CropLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::CropIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->axes) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["axis"] = value;

    value.clear();
    for (const auto& val : castedLayer->dim) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["dim"] = value;

    value.clear();
    for (const auto& val : castedLayer->offset) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["offset"] = value;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Maximum>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "max";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Divide>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "div";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Multiply>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "prod";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Squeeze>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Squeeze",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Squeeze>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v0::Unsqueeze>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Unsqueeze",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v0::Unsqueeze>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::DeformableConvolution>::createLayer(
        const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "DeformableConvolution",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::DeformableConvolutionLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::DeformableConvolution>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_pads_begin()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_pads_end()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    switch (castedLayer->get_auto_pad()) {
        case ngraph::op::PadType::SAME_UPPER:
            res->params["auto_pad"] = "same_upper";
            break;
        case ngraph::op::PadType::SAME_LOWER:
            res->params["auto_pad"] = "same_lower";
            break;
        case ngraph::op::PadType::VALID:
            res->params["auto_pad"] = "valid";
            break;
        default:
            break;
    }

    value.clear();
    for (const auto& val : castedLayer->get_strides()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_dilations()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["dilations"] = value;

    // Restore kernel size and output
    const auto& shape = castedLayer->get_input_shape(2);
    res->params["output"] = asString(shape[0]);

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty()) value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;

    res->params["group"] = asString(castedLayer->get_group());
    res->params["deformable_group"] = asString(castedLayer->get_deformable_group());

    const auto weightsNode = castedLayer->input_value(2).get_node_shared_ptr();
    InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::MaxPool>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Pooling",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PoolingLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::MaxPool>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_pads_begin()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["pads_begin"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_pads_end()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["pads_end"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_strides()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["strides"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_kernel()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["kernel"] = value;
    res->params["pool-method"] = "max";

    switch (castedLayer->get_auto_pad()) {
    case ngraph::op::PadType::VALID:
        res->params["auto_pad"] = "valid";
        break;
    case ngraph::op::PadType::SAME_UPPER:
        res->params["auto_pad"] = "same_upper";
        break;
    case ngraph::op::PadType::SAME_LOWER:
        res->params["auto_pad"] = "same_lower";
        break;
    default:
        break;
    }

    switch (castedLayer->get_rounding_type()) {
    case ngraph::op::RoundingType::CEIL:
        res->params["rounding_type"] = "ceil";
        break;
    case ngraph::op::RoundingType::FLOOR:
        res->params["rounding_type"] = "floor";
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported ngraph rounding type.";
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PSROIPooling>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "PSROIPooling",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PSROIPooling>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["output_dim"] = asString(castedLayer->get_output_dim());
    res->params["group_size"] = asString(castedLayer->get_group_size());
    res->params["spatial_bins_x"] = asString(castedLayer->get_spatial_bins_x());
    res->params["spatial_bins_y"] = asString(castedLayer->get_spatial_bins_y());
    res->params["spatial_scale"] = asString(castedLayer->get_spatial_scale());
    res->params["mode"] = castedLayer->get_mode();

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::DeformablePSROIPooling>::createLayer(
        const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "PSROIPooling",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::DeformablePSROIPooling>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["output_dim"] = asString(castedLayer->get_output_dim());
    res->params["group_size"] = asString(castedLayer->get_group_size());
    res->params["spatial_bins_x"] = asString(castedLayer->get_spatial_bins_x());
    res->params["spatial_bins_y"] = asString(castedLayer->get_spatial_bins_y());
    res->params["spatial_scale"] = asString(castedLayer->get_spatial_scale());
    res->params["mode"] = castedLayer->get_mode();
    res->params["trans_std"] = asString(castedLayer->get_trans_std());
    res->params["part_size"] = asString(castedLayer->get_part_size());
    res->params["no_trans"] = layer->get_input_size() == 2 ? "1" : "0";

    // temporary workaround due to incorrect usage of group_size in the nGraph operation for the DeformablePSROIPooling
    res->params["pooled_height"] = asString(castedLayer->get_group_size());
    res->params["pooled_width"] = asString(castedLayer->get_group_size());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::VariadicSplit>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Split",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::SplitLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::VariadicSplit>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto axis_node = castedLayer->input_value(1).get_node_shared_ptr();
    const auto axis_node_const = std::dynamic_pointer_cast<ngraph::op::Constant>(axis_node);
    if (!axis_node_const) {
        THROW_IE_EXCEPTION << "Split " << castedLayer->get_friendly_name() << " has no axes as Constant";
    }
    auto axis = axis_node_const->cast_vector<int64_t>()[0];
    if (axis < 0) {
        axis += castedLayer->get_input_shape(0).size();
    }
    res->params["axis"] = asString(axis);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Concat>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Concat",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ConcatLayer>(params);

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Concat>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->get_concatenation_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GatherIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Gather",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::GatherLayer>(params);

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::GatherIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->get_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ReverseSequence>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReverseSequence", details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReverseSequenceLayer>(params);

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ReverseSequence>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["batch_axis"] = asString(castedLayer->get_batch_axis());
    res->params["seq_axis"] = asString(castedLayer->get_sequence_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ShapeOf>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ShapeOf",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
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
CNNLayer::Ptr NodeConverter<ngraph::op::SquaredDifference>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "squared_diff";
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
CNNLayer::Ptr NodeConverter<ngraph::op::PowerIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Power",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PowerLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PowerIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["power"] = asString(castedLayer->power);
    res->params["scale"] = asString(castedLayer->scale);
    res->params["shift"] = asString(castedLayer->shift);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Eltwise>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Eltwise>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string type;
    switch (castedLayer->eltwise_type) {
    case ELTWISE_TYPE::Sum:
        type = "sum";
        break;
    case ELTWISE_TYPE::Sub:
        type = "sub";
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
CNNLayer::Ptr NodeConverter<ngraph::op::FullyConnected>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "FullyConnected",
                          details::convertPrecision(layer->get_output_element_type(0))};

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::FullyConnected>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::FullyConnectedLayer>(params);
    res->params["out-size"] = asString(castedLayer->get_out_size());

    auto & rt_info = layer->get_rt_info();
    bool keep_constants(false);
    if (auto attr = std::dynamic_pointer_cast<ngraph::VariantWrapper<int64_t>>(rt_info["keep_constants"])) {
        keep_constants = attr->get();
    }

    const auto weightsNode = layer->input_value(1).get_node_shared_ptr();
    if (!keep_constants && InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights)) {
        const auto biasNode = layer->input_value(2).get_node_shared_ptr();
        InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);
    }

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

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::RegionYolo>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "RegionYolo",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::RegionYolo>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_mask()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["mask"] = value;

    value = "";
    for (const auto& val : castedLayer->get_anchors()) {
        if (!value.empty())
            value += ",";
        value += asString(val);
    }
    res->params["anchors"] = value;

    res->params["coords"] = asString(castedLayer->get_num_coords());
    res->params["classes"] = asString(castedLayer->get_num_classes());
    res->params["num"] = asString(castedLayer->get_num_regions());
    res->params["do_softmax"] = castedLayer->get_do_softmax() ? "1" : "0";
    res->params["axis"] = asString(castedLayer->get_axis());
    res->params["end_axis"] = asString(castedLayer->get_end_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ReorgYolo>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReorgYolo",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ReorgYolo>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_strides()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }

    res->params["stride"] = value;
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Log>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Log",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::NormalizeIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Normalize",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::NormLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::NormalizeIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["eps"] = asString(castedLayer->get_eps());
    res->params["channel_shared"] = castedLayer->get_channel_shared() ? "1" : "0";
    res->params["across_spatial"] = castedLayer->get_across_spatial() ? "1" : "0";

    const auto weightsNode = layer->input_value(1).get_node_shared_ptr();
    if (auto constWeights = ngraph::as_type_ptr<ngraph::op::Constant>(weightsNode)) {
        Blob::Ptr dataBlob = InferenceEngine::details::shareWeights(constWeights);
        res->blobs["weights"] = dataBlob;
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Erf>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Erf",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sign>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sign",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sin>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sin",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sinh>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sinh",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Asin>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Asin",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Cos>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Cos",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Cosh>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Cosh",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Acos>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Acos",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Tan>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Tan",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Atan>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Atan",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sqrt>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sqrt",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

}  // namespace Builder
}  // namespace InferenceEngine
