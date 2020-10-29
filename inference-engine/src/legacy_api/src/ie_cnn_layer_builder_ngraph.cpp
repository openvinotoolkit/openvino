// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <cmath>
#include <set>
#include <sstream>
#include <utility>

#include "legacy/ngraph_ops/crop_ie.hpp"
#include "ngraph_ops/convolution_ie.hpp"
#include "legacy/ngraph_ops/eltwise.hpp"
#include "legacy/ngraph_ops/fully_connected.hpp"
#include "legacy/ngraph_ops/gather_ie.hpp"
#include "legacy/ngraph_ops/gather_tree_ie.hpp"
#include "legacy/ngraph_ops/gru_cell_ie.hpp"
#include "legacy/ngraph_ops/interp.hpp"
#include "legacy/ngraph_ops/lrn_ie.hpp"
#include "legacy/ngraph_ops/lstm_cell_ie.hpp"
#include <transformations/rt_info/primitives_priority_attribute.hpp>
#include "legacy/ngraph_ops/normalize_ie.hpp"
#include "legacy/ngraph_ops/nms_ie.hpp"
#include "legacy/ngraph_ops/onehot_ie.hpp"
#include "legacy/ngraph_ops/pad_ie.hpp"
#include "legacy/ngraph_ops/power.hpp"
#include "legacy/ngraph_ops/prior_box_clustered_ie.hpp"
#include "legacy/ngraph_ops/prior_box_ie.hpp"
#include "legacy/ngraph_ops/proposal_ie.hpp"
#include "legacy/ngraph_ops/relu_ie.hpp"
#include "legacy/ngraph_ops/selu_ie.hpp"
#include "legacy/ngraph_ops/scaleshift.hpp"
#include "legacy/ngraph_ops/tile_ie.hpp"
#include "legacy/ngraph_ops/rnn_cell_ie.hpp"
#include "legacy/ngraph_ops/hard_sigmoid_ie.hpp"
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

    std::map<uint64_t, std::vector<std::pair<std::string, uint64_t>>> ngraph_parameter_id_to_ie_layer_port;
    std::map<std::pair<std::string, uint64_t>, uint64_t> ie_layer_port_to_tensor_iterator_input_id;

    // inputs/outputs of TensorIterator body (ie)
    std::map<std::string, DataPtr> in_info_map;
    std::map<std::string, DataPtr> out_info_map;

    // inputs/outputs of TensorIterator (ngraph representation)
    auto parameters = tensor_iterator->get_function()->get_parameters();
    auto results = tensor_iterator->get_function()->get_results();

    // Convert body (ngraph representation) to CNNNetwork.
    // This network will contain nodes of type = "Input" and data nodes with wrong names.
    // IE TensorIterator doesn't include such nodes so we create CNNNetwork in a separate scope
    // to call the destructor and delete these "Input"/data nodes.

    // These layers will hold the necessary subnet after destruction of CNNNetwork.
    std::set<InferenceEngine::CNNLayerPtr> body_input_layers;
    // This map will save information about data nodes
    std::map<std::string, std::vector<TensorDesc>> layer_name_to_tensor_desc;
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
        InputsDataMap in_info_map_with_parameters;
        in_info_map_with_parameters = net.getInputsInfo();
        out_info_map = net.getOutputsInfo();

        // Fill the map to get layer and port of the body by the parameter index.
        uint64_t counter = 0;
        for (const auto& param : parameters) {
            auto info = in_info_map_with_parameters.at(param->get_friendly_name());
            auto data_ptr = info->getInputData();
            auto input_to = getInputTo(data_ptr);
            for (const auto& next_layer : input_to) {
                auto port_idx = find_input_idx(next_layer.second, data_ptr);
                ngraph_parameter_id_to_ie_layer_port[counter].push_back({next_layer.first, port_idx});
            }
            counter++;
        }

        // Temporary body to call deep copy
        InferenceEngine::TensorIterator::Body temp_body;
        for (const auto& in : in_info_map_with_parameters) {
            temp_body.inputs.emplace_back(in.second->getInputData());
        }

        for (const auto& out : out_info_map) {
            temp_body.outputs.emplace_back(out.second);
        }

        // This deep copy will hold all unreachable constants. See the comment in CopyTIBody function.
        auto deep_cp_body = InferenceEngine::NetPass::CopyTIBody(temp_body);
        for (const auto& data_ptr : deep_cp_body.inputs) {
            auto input_to = getInputTo(data_ptr);
            for (const auto& node : input_to) {
                // Make it compatible with ir v7: delete Input layers in body
                if (node.second->type != "Input") {
                    body_input_layers.emplace(node.second);
                    // Save information about data nodes to re-create them with correct names.
                    for (const auto& data : node.second->insData) {
                        layer_name_to_tensor_desc[node.second->name].emplace_back(data.lock()->getTensorDesc());
                    }
                }
            }
        }

        for (const auto& data_ptr : deep_cp_body.outputs) {
            out_info_map[data_ptr->getName()] = data_ptr;
        }
    }

    auto holder = std::make_shared<Data>("const_holder", Precision::UNSPECIFIED);
    for (const auto& input_layer : body_input_layers) {
        // Save all constants to the holder so that they are not deleted.
        if (input_layer->insData.empty()) {
            getInputTo(holder)[input_layer->name] = input_layer;
            continue;
        }

        // Re-create the data nodes with the correct names and fill inputs of TensorIterator (ie)
        for (size_t i = 0; i < input_layer->insData.size(); i++) {
            if (!input_layer->insData[i].lock()) {
                std::string data_name = (input_layer->insData.size() == 1)
                                            ? input_layer->name
                                            : input_layer->name + "." + std::to_string(i);

                DataPtr data(new Data(data_name, layer_name_to_tensor_desc[input_layer->name][i]));
                input_layer->insData[i] = data;
                getInputTo(data)[input_layer->name] = input_layer;
                in_info_map[data_name] = data;
            }
        }
    }

    // Create Inference Engine representation of TensorIterator
    LayerParams params = {layer->get_friendly_name(), "TensorIterator",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TensorIterator>(params);

    // Body: inputs
    uint64_t counter = 0;
    for (const auto& in : in_info_map) {
        res->body.inputs.emplace_back(in.second);

        // Fill the map to get the input index by layer and port of the body.
        auto input_to = getInputTo(in.second);
        for (const auto& next_layer : input_to) {
            auto port_idx = find_input_idx(next_layer.second, in.second);
            ie_layer_port_to_tensor_iterator_input_id[{next_layer.first, port_idx}] = counter;
        }
        counter++;
    }
    // the holder should be the last input element.
    res->body.inputs.emplace_back(holder);

    // Body: outputs
    for (const auto& out : out_info_map) {
        res->body.outputs.emplace_back(out.second);
    }

    // Port map: outputs
    for (const auto& desc : tensor_iterator->get_output_descriptions()) {
        auto result = results[desc->m_body_value_index]->input(0).get_source_output();

        std::string name = result.get_node()->get_friendly_name();
        if (result.get_node()->get_output_size() > 1) {
            name += "." + std::to_string(result.get_index());
        }
        auto output_layer = out_info_map.at(name);

        // Find index in outputs of the IE TensorIterator body
        auto it = std::find(res->body.outputs.begin(), res->body.outputs.end(), output_layer);
        if (it == res->body.outputs.end()) {
            THROW_IE_EXCEPTION << "Output layer not found.";
        }
        auto body_output_idx = it - res->body.outputs.begin();

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
        for (const auto& mapping : ngraph_parameter_id_to_ie_layer_port[desc->m_body_parameter_index]) {
            auto body_input_index = ie_layer_port_to_tensor_iterator_input_id.at(mapping);
            std::string type_name = desc->get_type_info().name;

            if (type_name == "SliceInputDescription") {
                auto input_desc = ::ngraph::as_type_ptr<ngraph::op::TensorIterator::SliceInputDescription>(desc);
                IE_ASSERT(input_desc != nullptr);

                res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                    static_cast<int>(input_desc->m_input_index), static_cast<int>(body_input_index),
                    static_cast<int>(input_desc->m_axis), static_cast<int>(input_desc->m_stride),
                    static_cast<int>(input_desc->m_start), static_cast<int>(input_desc->m_end),
                    static_cast<int>(input_desc->m_part_size)});
            } else if (type_name == "MergedInputDescription") {
                auto input_desc = ::ngraph::as_type_ptr<ngraph::op::TensorIterator::MergedInputDescription>(desc);
                IE_ASSERT(input_desc != nullptr);

                res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                    static_cast<int>(input_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});

                auto result = results[input_desc->m_body_value_index]->inputs()[0].get_source_output();

                // Create correct name for output.
                std::string output_name = result.get_node()->get_friendly_name();
                if (result.get_node()->get_output_size() > 1) {
                    output_name += "." + std::to_string(result.get_index());
                }

                auto output_layer = out_info_map.at(output_name);
                // Find index in outputs of the IE TensorIterator body
                auto it = std::find(res->body.outputs.begin(), res->body.outputs.end(), output_layer);
                if (it == res->body.outputs.end()) {
                    THROW_IE_EXCEPTION << "Output layer not found.";
                }
                auto body_output_idx = it - res->body.outputs.begin();

                res->back_edges.emplace_back(InferenceEngine::TensorIterator::PortMap {
                    static_cast<int>(body_output_idx), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
            } else if (type_name == "InvariantInputDescription") {
                auto input_desc = ::ngraph::as_type_ptr<ngraph::op::TensorIterator::InvariantInputDescription>(desc);
                IE_ASSERT(input_desc != nullptr);

                res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                        static_cast<int>(input_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
            } else {
                THROW_IE_EXCEPTION << "Incorrect type of the input description.";
            }
        }
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
CNNLayer::Ptr NodeConverter<ngraph::op::Constant>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Const",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Constant>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->blobs["custom"] = shareWeights(castedLayer);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Convert>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Convert",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto p = details::convertPrecision(layer->get_output_element_type(0));
    std::string precision_str;
    switch (p) {
    case Precision::FP16:
        precision_str = "FP16";
        break;
    case Precision::FP32:
        precision_str = "FP32";
        break;
    case Precision::I8:
        precision_str = "I8";
        break;
    case Precision::I16:
        precision_str = "I16";
        break;
    case Precision::I32:
        precision_str = "I32";
        break;
    case Precision::I64:
        precision_str = "I64";
        break;
    case Precision::U8:
        precision_str = "U8";
        break;
    case Precision::U16:
        precision_str = "U16";
        break;
    case Precision::U32:
        precision_str = "U32";
        break;
    case Precision::U64:
        precision_str = "U64";
        break;
    case Precision::BOOL:
        precision_str = "BOOL";
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported type";
    }

    res->params["precision"] = precision_str;
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
CNNLayer::Ptr NodeConverter<ngraph::op::Relu>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReLU",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReLULayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::SeluIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Selu",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::SeluIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["alpha"] = asString(castedLayer->alpha);
    res->params["gamma"] = asString(castedLayer->gamma);
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
CNNLayer::Ptr NodeConverter<ngraph::op::MVN>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "MVN", details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::MVNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::MVN>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["eps"] = asString(castedLayer->get_eps());

    const size_t chanelAxis = 1;
    ngraph::AxisSet reductionAxes = castedLayer->get_reduction_axes();
    res->params["across_channels"] = asString(reductionAxes.count(chanelAxis) > 0);

    res->params["normalize_variance"] = asString(castedLayer->get_normalize_variance());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::LRN_IE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Norm",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::NormLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::LRN_IE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["alpha"] = asString(castedLayer->get_alpha());
    res->params["beta"] = asString(castedLayer->get_beta());
    res->params["k"] = asString(castedLayer->get_bias());
    res->params["local-size"] = asString(castedLayer->get_nsize());
    res->params["region"] = castedLayer->get_region();
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
CNNLayer::Ptr NodeConverter<ngraph::op::Clamp>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Clamp",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ClampLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Clamp>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["min"] = asString(castedLayer->get_min());
    res->params["max"] = asString(castedLayer->get_max());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Softmax>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "SoftMax",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::SoftMaxLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::Softmax>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->get_axis());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Subtract>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "sub";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Power>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "pow";
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
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Minimum>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "min";
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
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Add>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "sum";
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
CNNLayer::Ptr NodeConverter<ngraph::op::Unsqueeze>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Unsqueeze",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Unsqueeze>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::FakeQuantize>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "FakeQuantize",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::QuantizeLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::FakeQuantize>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;
    res->params["levels"] = asString(castedLayer->get_levels());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ConvolutionIE>::createLayer(
        const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Convolution",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ConvolutionLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ConvolutionIE>(layer);
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
    const auto& shape = castedLayer->get_input_shape(1);
    res->params["output"] = asString(castedLayer->get_shape()[1]);
    res->params["group"] = asString(castedLayer->get_group());

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty()) value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;

    auto & rt_info = layer->get_rt_info();
    bool keep_constants(false);
    if (auto attr = std::dynamic_pointer_cast<ngraph::VariantWrapper<int64_t>>(rt_info["keep_constants"])) {
        keep_constants = attr->get();
    }

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = castedLayer->input_value(1).get_node_shared_ptr();
    if (!keep_constants && converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];

        if (castedLayer->inputs().size() == 3) {
            const auto biasNode = castedLayer->input_value(2).get_node_shared_ptr();
            if (converter.canCreate(biasNode)) {
                const auto& bias = converter.createLayer(biasNode);
                res->blobs["biases"] = bias->blobs["custom"];
                res->_biases = bias->blobs["custom"];
            }
        }
    }
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

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = castedLayer->input_value(2).get_node_shared_ptr();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::AvgPool>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Pooling",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PoolingLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::AvgPool>(layer);
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

    auto exclude_pad = castedLayer->get_exclude_pad();
    res->params["exclude-pad"] = exclude_pad ? "true" : "false";
    res->params["pool-method"] = "avg";
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
CNNLayer::Ptr NodeConverter<ngraph::op::ROIPooling>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ROIPooling",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ROIPooling>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["pooled_h"] = asString(castedLayer->get_output_size()[0]);
    res->params["pooled_w"] = asString(castedLayer->get_output_size()[1]);
    res->params["spatial_scale"] = asString(castedLayer->get_spatial_scale());
    res->params["method"] = castedLayer->get_method();

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
CNNLayer::Ptr NodeConverter<ngraph::op::PRelu>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "PReLU",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PReLULayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PRelu>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    const auto weightsNode = castedLayer->input_value(1).get_node_shared_ptr();
    if (auto const_weights = ngraph::as_type_ptr<ngraph::op::Constant>(weightsNode)) {
        SizeVector dataShape = const_weights->get_shape();
        if (dataShape.size() >= 2 && ngraph::shape_size(dataShape) == dataShape[1]) {
            dataShape = {dataShape[1]};
        }

        Blob::Ptr dataBlb = shareWeights(const_weights);

        res->blobs["weights"] = dataBlb;
        res->_weights = dataBlb;
    }

    auto const_shape = castedLayer->input(1).get_shape(), tensor_shape = castedLayer->input(0).get_shape();
    if (const_shape.size() == 1 && const_shape[0] == 1) {
        res->params["channel_shared"] = "true";
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Split>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Split",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::SplitLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::Split>(layer);
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
CNNLayer::Ptr NodeConverter<ngraph::op::GatherTreeIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "GatherTree",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
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
CNNLayer::Ptr NodeConverter<ngraph::op::Reshape>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Reshape",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReshapeLayer>(params);
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
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Reshape>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Reshape",
                          details::convertPrecision(layer->get_output_element_type(0))};

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::Reshape>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;


    const auto constNode = castedLayer->input_value(1).get_node_shared_ptr();
    if (auto constValue = ngraph::as_type_ptr<ngraph::op::Constant>(constNode)) {
        auto value = constValue->cast_vector<int64_t>();
        for (auto & i : value) {
            if (i == 0 && !castedLayer->get_special_zero())
                THROW_IE_EXCEPTION << "Reshape " << params.name << " has `special_zero`=False and zeros in second input. This combination is not supported";
        }
    } else {
        THROW_IE_EXCEPTION << "Reshape " << params.name << " has dynamic second input!";
    }

    auto res = std::make_shared<InferenceEngine::ReshapeLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PadIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Pad",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PadLayer>(params);

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PadIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    switch (castedLayer->get_pad_mode()) {
    case ngraph::op::PadMode::EDGE:
        res->params["pad_mode"] = "edge";
        break;
    case ngraph::op::PadMode::REFLECT:
        res->params["pad_mode"] = "reflect";
        break;
    case ngraph::op::PadMode::CONSTANT:
        res->params["pad_mode"] = "constant";
        res->params["pad_value"] = asString(castedLayer->get_pad_value());
        break;
    case ngraph::op::PadMode::SYMMETRIC:
        res->params["pad_mode"] = "symmetric";
    }
    std::string pad;
    for (const auto& p : castedLayer->get_pads_begin()) {
        if (!pad.empty()) pad += ",";
        pad += asString(p);
    }
    res->params["pads_begin"] = pad;

    pad.clear();
    for (const auto& p : castedLayer->get_pads_end()) {
        if (!pad.empty()) pad += ",";
        pad += asString(p);
    }
    res->params["pads_end"] = pad;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ScaleShiftIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ScaleShift",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ScaleShiftLayer>(params);

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = layer->input_value(1).get_node_shared_ptr();
    if (converter.canCreate(weightsNode)) {
        const auto& weightsLayer = converter.createLayer(weightsNode);
        res->blobs["weights"] = weightsLayer->blobs["custom"];
        res->_weights = weightsLayer->blobs["custom"];
    }

    const auto biasNode = layer->input_value(2).get_node_shared_ptr();
    if (converter.canCreate(biasNode)) {
        const auto& bias = converter.createLayer(biasNode);
        res->blobs["biases"] = bias->blobs["custom"];
        res->_biases = bias->blobs["custom"];
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Elu>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "elu",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Elu>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["alpha"] = asString(castedLayer->get_alpha());

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
CNNLayer::Ptr NodeConverter<ngraph::op::ProposalIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Proposal",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ProposalIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attr = castedLayer->get_attrs();
    std::string param;
    for (const auto& val : attr.ratio) {
        if (!param.empty()) param += ",";
        param += asString(val);
    }
    res->params["ratio"] = param;

    param.clear();
    for (const auto& val : attr.scale) {
        if (!param.empty()) param += ",";
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
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBoxClusteredIE>::createLayer(
    const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "PriorBoxClustered",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PriorBoxClusteredIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attr = castedLayer->get_attrs();
    std::string param;
    for (const auto& val : attr.widths) {
        if (!param.empty()) param += ",";
        param += asString(val);
    }
    res->params["width"] = param;

    param.clear();
    for (const auto& val : attr.heights) {
        if (!param.empty()) param += ",";
        param += asString(val);
    }
    res->params["height"] = param;

    param.clear();
    for (const auto& val : attr.variances) {
        if (!param.empty()) param += ",";
        param += asString(val);
    }
    res->params["variance"] = param;

    if (std::abs(attr.step_heights - attr.step_widths) < 1e-5) {
        res->params["step"] = asString(attr.step_widths);
    } else {
        res->params["step_w"] = asString(attr.step_widths);
        res->params["step_h"] = asString(attr.step_heights);
    }
    res->params["offset"] = asString(attr.offset);
    res->params["clip"] = asString(attr.clip ? 1 : 0);
    res->params["flip"] = "1";

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBoxIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "PriorBox",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PriorBoxIE>(layer);
    auto layer_info = params.type + " layer " + params.name;

    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << layer_info;

    auto attr = castedLayer->get_attrs();
    std::string param;

    auto data_pshape = castedLayer->get_input_partial_shape(0);
    if (data_pshape.is_dynamic()) THROW_IE_EXCEPTION << "Dynamic 0-port input of " << layer_info << " is not supported";
    auto data_shape = data_pshape.to_shape();
    if (data_shape.size() != 4) THROW_IE_EXCEPTION << layer_info << " has " << data_shape.size() << " items in 0-port input, 4 expected";

    auto img_pshape = castedLayer->get_input_partial_shape(1);
    if (img_pshape.is_dynamic()) THROW_IE_EXCEPTION << "Dynamic 1-port input of " << layer_info << " is not supported";
    auto img_shape = img_pshape.to_shape();
    if (img_shape.size() != 4) THROW_IE_EXCEPTION << layer_info << " has " << data_shape.size() << " items in 1-port input, 4 expected";

    if (!attr.scale_all_sizes) {
        // mxnet-like PriorBox
        auto img_H = img_shape[2];
        auto data_H = data_shape[2];
        if (attr.step == -1)
            attr.step = static_cast<float>(1. * img_H / data_H);
        else
            attr.step *= img_H;
        for (auto& size : attr.min_size)
            size *= img_H;
    }

    for (const auto& val : attr.max_size) {
        if (!param.empty()) param += ",";
        param += asString(val);
    }
    res->params["max_size"] = param;

    param.clear();
    for (const auto& val : attr.min_size) {
        if (!param.empty()) param += ",";
        param += asString(val);
    }
    res->params["min_size"] = param;

    param.clear();
    for (const auto& val : attr.aspect_ratio) {
        if (!param.empty()) param += ",";
        param += asString(val);
    }
    res->params["aspect_ratio"] = param;

    param.clear();
    for (const auto& val : attr.variance) {
        if (!param.empty()) param += ",";
        param += asString(val);
    }
    res->params["variance"] = param;

    res->params["step"] = asString(attr.step);
    res->params["offset"] = asString(attr.offset);
    res->params["clip"] = asString(attr.clip ? 1 : 0);
    res->params["flip"] = asString(attr.flip ? 1 : 0);
    res->params["scale_all_sizes"] = asString(attr.scale_all_sizes ? 1 : 0);

    res->params["density"] = asString(attr.density);
    res->params["fixed_size"] = asString(attr.fixed_size);
    res->params["fixed_ratio"] = asString(attr.fixed_ratio);

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
CNNLayer::Ptr NodeConverter<ngraph::op::TileIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Tile",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TileLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::TileIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->axis);
    res->params["tiles"] = asString(castedLayer->tiles);

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
CNNLayer::Ptr NodeConverter<ngraph::op::Interp>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Resample",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Interp>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attrs = castedLayer->get_attrs();

    if (attrs.antialias) {
        THROW_IE_EXCEPTION << "Interp do not support antialias";
    }
    if (attrs.mode != "linear") {
        THROW_IE_EXCEPTION << "Interp do not support mode '" << attrs.mode << "'";
    }

    params = {layer->get_friendly_name(), "Interp",
              details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    res->params["height"] = asString(attrs.height);
    res->params["width"] = asString(attrs.width);
    res->params["pad_beg"] = asString(attrs.pad_beg);
    res->params["pad_end"] = asString(attrs.pad_end);
    res->params["align_corners"] = attrs.align_corners ? "1" : "0";

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v0::Interpolate>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "Interpolate operation should be converted to Interp";
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

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = layer->input_value(1).get_node_shared_ptr();
    if (!keep_constants && converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];

        const auto biasNode = layer->input_value(2).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::MatMul>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Gemm",
                          details::convertPrecision(layer->get_output_element_type(0))};

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::MatMul>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::GemmLayer>(params);
    res->params["transpose_a"] = castedLayer->get_transpose_a() ? "True" : "False";
    res->params["transpose_b"] = castedLayer->get_transpose_b() ? "True" : "False";

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

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = castedLayer->input_value(1).get_node_shared_ptr();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
    } else {
        THROW_IE_EXCEPTION << "Cannot convert weight node for NormalizeIE op";
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::CTCGreedyDecoder>::createLayer(
    const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "CTCGreedyDecoder",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::CTCGreedyDecoder>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["ctc_merge_repeated"] = castedLayer->get_ctc_merge_repeated() ? "1" : "0";
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

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::OneHotIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "OneHot", Precision::FP32};

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::OneHotIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::OneHotLayer>(params);
    res->params["axis"] = std::to_string(castedLayer->get_axis());
    res->params["depth"] = std::to_string(castedLayer->get_depth());
    res->params["on_value"] = std::to_string(castedLayer->get_on_value());
    res->params["off_value"] = std::to_string(castedLayer->get_off_value());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::HardSigmoid_IE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = { layer->get_friendly_name(), "HardSigmoid", details::convertPrecision(layer->get_output_element_type(0)) };
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::HardSigmoid_IE>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["alpha"] = asString(castedLayer->get_alpha());
    res->params["beta"] = asString(castedLayer->get_beta());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GRN>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "GRN",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::GRN>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::GRNLayer>(params);
    res->params["bias"] = asString(castedLayer->get_bias());
    return res;
}

}  // namespace Builder
}  // namespace InferenceEngine
