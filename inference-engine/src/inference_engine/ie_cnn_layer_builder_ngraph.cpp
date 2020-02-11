// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_cnn_layer_builder_ngraph.h>
#include "cnn_network_ngraph_impl.hpp"
#include <ie_layer_parsers.h>
#include <precision_utils.h>


#include <limits>
#include <cmath>
#include <ngraph/ngraph.hpp>
#include <ngraph/type.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/abs.hpp>
#include <ngraph/op/acos.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/op/asin.hpp>
#include <ngraph/op/atan.hpp>
#include <ngraph/op/avg_pool.hpp>
#include <ngraph/op/batch_norm.hpp>
#include <ngraph/op/broadcast.hpp>
#include <ngraph/op/ceiling.hpp>
#include <ngraph/op/concat.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/cos.hpp>
#include <ngraph/op/cosh.hpp>
#include <ngraph/op/deformable_convolution.hpp>
#include <ngraph/op/deformable_psroi_pooling.hpp>
#include <ngraph/op/divide.hpp>
#include <ngraph/op/exp.hpp>
#include <ngraph/op/experimental/dyn_reshape.hpp>
#include <ngraph/op/experimental/layers/ctc_greedy_decoder.hpp>
#include <ngraph/op/experimental/layers/detection_output.hpp>
#include <ngraph/op/experimental/layers/interpolate.hpp>
#include <ngraph/op/experimental/layers/prior_box.hpp>
#include <ngraph/op/experimental/layers/prior_box_clustered.hpp>
#include <ngraph/op/experimental/layers/proposal.hpp>
#include <ngraph/op/experimental/layers/psroi_pooling.hpp>
#include <ngraph/op/experimental/layers/region_yolo.hpp>
#include <ngraph/op/experimental/layers/reorg_yolo.hpp>
#include <ngraph/op/experimental/layers/roi_pooling.hpp>
#include <ngraph/op/experimental/range.hpp>
#include <ngraph/op/experimental/shape_of.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/floor.hpp>
#include <ngraph/op/fused/clamp.hpp>
#include <ngraph/op/fused/conv_fused.hpp>
#include <ngraph/op/fused/elu.hpp>
#include <ngraph/op/fused/group_conv.hpp>
#include <ngraph/op/fused/grn.hpp>
#include <ngraph/op/fused/hard_sigmoid.hpp>
#include <ngraph/op/fused/mvn.hpp>
#include <ngraph/op/fused/normalize_l2.hpp>
#include <ngraph/op/fused/prelu.hpp>
#include <ngraph/op/fused/split.hpp>
#include <ngraph/op/fused/squared_difference.hpp>
#include <ngraph/op/gather.hpp>
#include <ngraph/op/less.hpp>
#include <ngraph/op/log.hpp>
#include <ngraph/op/lrn.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/multiply.hpp>
#include <ngraph/op/non_max_suppression.hpp>
#include <ngraph/op/pad.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/power.hpp>
#include <ngraph/op/reduce_mean.hpp>
#include <ngraph/op/reduce_prod.hpp>
#include <ngraph/op/reduce_sum.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/reverse_sequence.hpp>
#include <ngraph/op/select.hpp>
#include <ngraph/op/sigmoid.hpp>
#include <ngraph/op/sin.hpp>
#include <ngraph/op/sinh.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/op/sqrt.hpp>
#include <ngraph/op/subtract.hpp>
#include <ngraph/op/tan.hpp>
#include <ngraph/op/tanh.hpp>
#include <ngraph/op/tensor_iterator.hpp>
#include <ngraph/shape.hpp>
#include <ngraph_ops/lstm_cell_ie.hpp>
#include <ngraph/op/and.hpp>
#include <ngraph/op/not.hpp>
#include <ngraph/op/or.hpp>
#include <ngraph/op/reduce_logical_and.hpp>
#include <ngraph/op/reduce_logical_or.hpp>
#include <ngraph/op/xor.hpp>
#include <set>
#include <sstream>
#include <utility>


#include "graph_tools.hpp"
#include "net_pass.h"
#include "ngraph_ops/crop_ie.hpp"
#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/fully_connected.hpp"
#include "ngraph_ops/gather_ie.hpp"
#include "ngraph_ops/gather_tree_ie.hpp"
#include "ngraph_ops/gemm.hpp"
#include "ngraph_ops/generic_ie.hpp"
#include "ngraph_ops/group_conv_bias.hpp"
#include "ngraph_ops/interp.hpp"
#include "ngraph_ops/lrn_ie.hpp"
#include "ngraph_ops/normalize_ie.hpp"
#include "ngraph_ops/nms_ie.hpp"
#include "ngraph_ops/onehot_ie.hpp"
#include "ngraph_ops/pad_ie.hpp"
#include "ngraph_ops/power.hpp"
#include "ngraph_ops/prior_box_clustered_ie.hpp"
#include "ngraph_ops/prior_box_ie.hpp"
#include "ngraph_ops/proposal_ie.hpp"
#include "ngraph_ops/quantize_conv_bias_fused.hpp"
#include "ngraph_ops/relu_ie.hpp"
#include "ngraph_ops/selu_ie.hpp"
#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/tensor_iterator.hpp"
#include "ngraph_ops/tile_ie.hpp"
#include "ngraph_ops/topk_ie.hpp"
#include "ngraph_ops/strided_slice_ie.hpp"
#include "ngraph_ops/hard_sigmoid_ie.hpp"

#include "ie_ngraph_utils.hpp"
#include "graph_transformer.h"

namespace InferenceEngine {
namespace Builder {

template <>
inline std::string INodeConverter::asString<double>(const double& value) {
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
inline std::string INodeConverter::asString<float>(const float& value) {
    return asString(static_cast<double>(value));
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Abs>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Abs",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GenericIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::GenericIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get layer " << layer->get_friendly_name();

    LayerParams params = {layer->get_friendly_name(), castedLayer->getType(),
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::TensorIterator>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
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

    auto tensor_iterator = ngraph::as_type_ptr<ngraph::op::TensorIterator>(layer);
    if (!tensor_iterator) {
        THROW_IE_EXCEPTION << "Cannot cast layer to TensorIterator.";
    }

    std::map<uint64_t, std::vector<std::pair<std::string, uint64_t>>> ngraph_parameter_id_to_ie_layer_port;
    std::map<std::pair<std::string, uint64_t>, uint64_t> ie_layer_port_to_tensor_iterator_input_id;

    // inputs/outputs of TensorIterator body (ie)
    std::map<std::string, DataPtr> in_info_map;
    std::map<std::string, DataPtr> out_info_map;

    // inputs/outputs of TensorIterator (ngraph representation)
    auto parameters = tensor_iterator->get_body()->get_parameters();
    auto results = tensor_iterator->get_body()->get_results();

    // Convert body (ngraph representation) to CNNNetwork.
    // This network will contain nodes of type = "Input" and data nodes with wrong names.
    // IE TensorIterator doesn't include such nodes so we create CNNNetwork in a separate scope
    // to call the destructor and delete these "Input"/data nodes.

    // These layers will hold the necessary subnet after destruction of CNNNetwork.
    std::set<InferenceEngine::CNNLayerPtr> body_input_layers;
    // This map will save information about data nodes
    std::map<std::string, std::vector<TensorDesc>> layer_name_to_tensor_desc;
    {
        auto tiBody = std::make_shared<details::TINGraphBody>(std::make_shared<ngraph::Function>(results, parameters));
        CNNNetwork net(tiBody);
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
            auto input_to = data_ptr->getInputTo();
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
            auto input_to = data_ptr->getInputTo();
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
            holder->getInputTo()[input_layer->name] = input_layer;
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
                data->getInputTo()[input_layer->name] = input_layer;
                in_info_map[data_name] = data;
            }
        }
    }

    // Create Inference Engine representation of TensorIterator
    LayerParams params = {layer->get_friendly_name(), "TensorIterator",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TensorIterator>(params);

    // Body: inputs
    uint64_t counter = 0;
    for (const auto& in : in_info_map) {
        res->body.inputs.emplace_back(in.second);

        // Fill the map to get the input index by layer and port of the body.
        auto input_to = in.second->getInputTo();
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
        auto result = results[desc->m_body_value_index]->inputs()[0].get_source_output();

        // GetOutputElement layer can be inserted by ngraph deep copy functions
        // (e.g. specialize_function, clone_function)
        // Take the previous layer.
        if (::ngraph::is_type<ngraph::op::GetOutputElement>(result.get_node_shared_ptr())) {
            result = result.get_node()->input(0).get_source_output();
        }
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

                // GetOutputElement layer can be inserted by ngraph deep copy functions
                // (e.g. specialize_function, clone_function)
                // Take the previous layer.
                if (::ngraph::is_type<ngraph::op::GetOutputElement>(result.get_node_shared_ptr())) {
                    result = result.get_node()->input(0).get_source_output();
                }
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

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Constant>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Const",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Constant>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->blobs["custom"] = shareWeights(castedLayer);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Convert>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Convert",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto p = details::ngraph::convertPrecision(layer->get_output_element_type(0));
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Floor>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Floor",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sigmoid>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sigmoid",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Tanh>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "TanH",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Relu>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReLU",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReLULayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::SeluIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Selu",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReLULayer>(params);

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::ReLUIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["negative_slope"] = asString(castedLayer->get_slope());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Range>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Range",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Exp>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Exp",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::MVN>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "MVN",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::MVNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::MVN>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["eps"] = asString(castedLayer->get_eps());
    if (castedLayer->get_reduction_axes().size() == castedLayer->get_shape().size()) {
        res->params["across_channels"] = "1";
    } else {
        res->params["across_channels"] = "0";
    }
    res->params["normalize_variance"] = asString(castedLayer->get_normalize_variance());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::LRN>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "LRN operation should be converted to LRN_IE";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::LRN_IE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Norm",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::SoftMaxLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::Softmax>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->get_axis());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Subtract>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "sub";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Power>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "pow";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Maximum>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "max";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Minimum>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "Minimum operation should be decomposed";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Divide>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "div";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Multiply>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "prod";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Add>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "sum";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Broadcast>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "Broadcast operation " << layer->get_friendly_name()
                       << " should be converted to Tile operation";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::BatchNormInference>::createLayer(
    const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "BatchNormInference operation should be fused or decomposed";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Squeeze>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Squeeze",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Squeeze>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Unsqueeze>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Unsqueeze",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Unsqueeze>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::FakeQuantize>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "FakeQuantize",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
    InferenceEngine::Parameter attr(rt_info["keep_constants"]);
    bool keep_constants = attr.as<bool>();

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = castedLayer->input_value(1).get_node_shared_ptr();
    if (!keep_constants && converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];

        if (castedLayer->inputs().size() == 3) {
            const auto biasNode = castedLayer->get_inputs()[2].get_output().get_node();
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
CNNLayer::Ptr NodeConverter<ngraph::op::DeconvolutionIE>::createLayer(
        const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Deconvolution",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::DeconvolutionLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::DeconvolutionIE>(layer);
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
    for (const auto& val : castedLayer->get_dilations()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["dilations"] = value;

    // Restore kernel size and output
    const auto& shape = castedLayer->get_input_shape(1);
    res->params["output"] = asString(shape[1]);

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty()) value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;
    res->params["group"] = asString(castedLayer->get_group());

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = castedLayer->input_value(1).get_node_shared_ptr();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];

        if (castedLayer->inputs().size() == 3) {
            const auto biasNode = castedLayer->get_inputs()[2].get_output().get_node();
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
CNNLayer::Ptr NodeConverter<ngraph::op::v1::BinaryConvolution>::createLayer(
        const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "BinaryConvolution",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::BinaryConvolutionLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::BinaryConvolution>(layer);
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
    res->params["output"] = asString(shape[0]);

    value.clear();
    for (size_t i = 2; i < shape.size(); i++) {
        if (!value.empty()) value += ",";
        value += asString(shape[i]);
    }
    res->params["kernel"] = value;

    switch (castedLayer->get_mode()) {
        case ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT:
            res->params["mode"] = "xnor-popcount";
    }

    auto weights_shape = castedLayer->input(1).get_source_output().get_shape();
    res->params["input"] = asString(weights_shape[1]);
    res->params["pad_value"] = asString(castedLayer->get_pad_value());

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::DeformableConvolution>::createLayer(
        const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "DeformableConvolution",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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

    auto & rt_info = layer->get_rt_info();
    InferenceEngine::Parameter attr(rt_info["keep_constants"]);
    bool keep_constants = attr.as<bool>();

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = castedLayer->input_value(2).get_node_shared_ptr();
    if (!keep_constants && converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::AvgPool>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Pooling",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PReLULayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PRelu>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    const auto weightsNode = castedLayer->input(1).get_source_output().get_node_shared_ptr();
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::SplitLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::Split>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto axis_node = castedLayer->input_value(1).get_node_shared_ptr();
    const auto axis_node_const = std::dynamic_pointer_cast<ngraph::op::Constant>(axis_node);
    if (!axis_node_const) {
        THROW_IE_EXCEPTION << "Split " << castedLayer->get_friendly_name() << " has no axes as Constant";
    }
    auto axis = axis_node_const->get_vector<int64_t>()[0];
    if (axis < 0) {
        axis += castedLayer->get_input_shape(0).size();
    }
    res->params["axis"] = asString(axis);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::VariadicSplit>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Split",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::SplitLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::VariadicSplit>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto axis_node = castedLayer->input_value(1).get_node_shared_ptr();
    const auto axis_node_const = std::dynamic_pointer_cast<ngraph::op::Constant>(axis_node);
    if (!axis_node_const) {
        THROW_IE_EXCEPTION << "Split " << castedLayer->get_friendly_name() << " has no axes as Constant";
    }
    auto axis = axis_node_const->get_vector<int64_t>()[0];
    if (axis < 0) {
        axis += castedLayer->get_input_shape(0).size();
    }
    res->params["axis"] = asString(axis);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Concat>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Concat",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ConcatLayer>(params);

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Concat>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->get_concatenation_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GatherIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Gather",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::GatherLayer>(params);

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::GatherIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->get_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::GatherTree>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "GatherTree operation should be converted to GatherTreeIE";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GatherTreeIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "GatherTree",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ReverseSequence>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReverseSequence", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReshapeLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ShapeOf>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ShapeOf",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::Reshape>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Reshape",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::Reshape>(layer);
    if (castedLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;


    const auto constNode = castedLayer->get_inputs()[1].get_output().get_node();
    if (auto constValue = ngraph::as_type_ptr<ngraph::op::Constant>(constNode)) {
        auto value = constValue->get_vector<int64_t>();
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ScaleShiftLayer>(params);

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weightsLayer = converter.createLayer(weightsNode);
        res->blobs["weights"] = weightsLayer->blobs["custom"];
        res->_weights = weightsLayer->blobs["custom"];
    }

    const auto biasNode = layer->get_inputs()[2].get_output().get_node();
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Elu>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["alpha"] = asString(castedLayer->get_alpha());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::SquaredDifference>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->params["operation"] = "squared_diff";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::DetectionOutput>::createLayer(
    const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "DetectionOutput",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::DetectionOutput>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attr = castedLayer->get_attrs();
    std::string param;

    res->params["num_classes"] = asString(attr.num_classes);
    res->params["background_label_id"] = asString(attr.background_label_id);
    res->params["top_k"] = asString(attr.top_k);
    res->params["variance_encoded_in_target"] = (attr.variance_encoded_in_target ? "1" : "0");
    for (const auto& val : attr.keep_top_k) {
        if (!param.empty()) param += ",";
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
CNNLayer::Ptr NodeConverter<ngraph::op::Transpose>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Permute",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    NodeConverter<ngraph::op::Constant> converter;
    const auto orderNode = layer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(orderNode)) {
        const auto& orderLayer = converter.createLayer(orderNode);
        auto order = orderLayer->blobs["custom"];
        int64_t* data = order->buffer().as<int64_t*>();
        std::string orderStr;
        for (size_t i = 0; i < order->size(); i++) {
            if (!orderStr.empty()) orderStr += ",";
            orderStr += asString(data[i]);
        }
        res->params["order"] = orderStr;
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Proposal>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "Proposal operation should be converted to ProposalIE";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ProposalIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Proposal",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBoxClustered>::createLayer(
    const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "PriorBoxClustered operation must be converted to PriorBoxClusteredIE operation.";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBoxIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "PriorBox",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PriorBoxIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto attr = castedLayer->get_attrs();
    std::string param;
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
CNNLayer::Ptr NodeConverter<ngraph::op::PriorBox>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "PriorBox operation must be converted to PriorBoxIE operation.";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::PowerIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Power",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::PowerLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::PowerIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["power"] = asString(castedLayer->power);
    res->params["scale"] = asString(castedLayer->scale);
    res->params["shift"] = asString(castedLayer->shift);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::TopK>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "TopK",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TopKLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::TopK>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto mode = castedLayer->get_mode();
    std::string str_mode;
    switch (mode) {
    case ngraph::op::v1::TopK::Mode::MIN:
        str_mode = "min";
        break;
    case ngraph::op::v1::TopK::Mode::MAX:
        str_mode = "max";
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported TopK mode";
    }

    auto sort = castedLayer->get_sort_type();
    std::string str_sort;
    switch (sort) {
    case ngraph::op::v1::TopK::SortType::NONE:
        str_sort = "none";
        break;
    case ngraph::op::v1::TopK::SortType::SORT_VALUES:
        str_sort = "value";
        break;
    case ngraph::op::v1::TopK::SortType::SORT_INDICES:
        str_sort = "index";
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported TopK sort type";
    }

    res->params["mode"] = str_mode;
    res->params["sort"] = str_sort;
    res->params["axis"] = asString(castedLayer->get_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::TopKIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "TopK",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TopKLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::TopKIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["mode"] = castedLayer->get_mode();
    res->params["sort"] = castedLayer->get_sort_type();
    res->params["axis"] = asString(castedLayer->get_axis());

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Eltwise>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Eltwise",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::Eltwise>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

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
CNNLayer::Ptr NodeConverter<ngraph::op::TileIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Tile",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TileLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::TileIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["axis"] = asString(castedLayer->axis);
    res->params["tiles"] = asString(castedLayer->tiles);

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::ResampleV2>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Resample", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
              details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);

    res->params["height"] = asString(attrs.height);
    res->params["width"] = asString(attrs.width);
    res->params["pad_beg"] = asString(attrs.pad_beg);
    res->params["pad_end"] = asString(attrs.pad_end);
    res->params["align_corners"] = attrs.align_corners ? "1" : "0";

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Interpolate>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "Interpolate operation should be converted to Interp";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::FullyConnected>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "FullyConnected",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::FullyConnected>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::FullyConnectedLayer>(params);
    res->params["out-size"] = asString(castedLayer->get_out_size());

    auto & rt_info = layer->get_rt_info();
    InferenceEngine::Parameter attr(rt_info["keep_constants"]);
    bool keep_constants = attr.as<bool>();

    NodeConverter<ngraph::op::Constant> converter;

    const auto weightsNode = layer->get_inputs()[1].get_output().get_node();
    if (!keep_constants && converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];

        const auto biasNode = layer->get_inputs()[2].get_output().get_node();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::LSTMCell>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "LSTMCell operation must be converted to LSTMCellIE operation.";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::LSTMCellIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "LSTMCell",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::LSTMCellIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::LSTMCell>(params);
    res->params["hidden_size"] = asString(castedLayer->get_hidden_size());
    std::string value;
    for (const auto& val : castedLayer->get_activations()) {
        if (!value.empty()) value += ",";
        value += val;
    }
    res->params["activations"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_activations_alpha()) {
        if (!value.empty()) value += ",";
        value += val;
    }
    res->params["activations_alpha"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_activations_beta()) {
        if (!value.empty()) value += ",";
        value += val;
    }
    res->params["activations_beta"] = value;
    res->params["clip"] = asString(castedLayer->get_clip());

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = layer->get_inputs()[3].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
        res->_weights = weights->blobs["custom"];
    }

    const auto biasNode = layer->get_inputs()[4].get_output().get_node();
    if (converter.canCreate(biasNode)) {
        const auto& bias = converter.createLayer(biasNode);
        res->blobs["biases"] = bias->blobs["custom"];
        res->_biases = bias->blobs["custom"];
    }
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::GemmIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Gemm",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};

    auto castedLayer = ngraph::as_type_ptr<ngraph::op::GemmIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::GemmLayer>(params);
    res->params["transpose_a"] = castedLayer->get_transpose_a() ? "True" : "False";
    res->params["transpose_b"] = castedLayer->get_transpose_b() ? "True" : "False";

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::RegionYolo>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "RegionYolo",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
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
CNNLayer::Ptr NodeConverter<ngraph::op::v1::ReduceMin>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReduceMin",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReduceLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::ReduceMin>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["keep_dims"] = castedLayer->get_keep_dims() ? "true" : "false";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::ReduceMax>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReduceMax",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReduceLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::ReduceMax>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["keep_dims"] = castedLayer->get_keep_dims() ? "true" : "false";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::ReduceMean>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReduceMean",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReduceLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::ReduceMean>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["keep_dims"] = castedLayer->get_keep_dims() ? "true" : "false";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::ReduceProd>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReduceProd",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReduceLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::ReduceProd>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["keep_dims"] = castedLayer->get_keep_dims() ? "true" : "false";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::ReduceSum>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReduceSum",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReduceLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::v1::ReduceSum>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["keep_dims"] = castedLayer->get_keep_dims() ? "true" : "false";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::NormalizeL2>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "NormalizeL2 operation should be converted to NormalizeIE";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Log>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Log",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::NormalizeIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Normalize",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::NormLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::NormalizeIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["eps"] = asString(castedLayer->get_eps());
    res->params["channel_shared"] = castedLayer->get_channel_shared() ? "1" : "0";
    res->params["across_spatial"] = castedLayer->get_across_spatial() ? "1" : "0";

    NodeConverter<ngraph::op::Constant> converter;
    const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
    if (converter.canCreate(weightsNode)) {
        const auto& weights = converter.createLayer(weightsNode);
        res->blobs["weights"] = weights->blobs["custom"];
    }

    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::CTCGreedyDecoder>::createLayer(
    const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "CTCGreedyDecoder",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    auto castedLayer = ngraph::as_type_ptr<ngraph::op::CTCGreedyDecoder>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["ctc_merge_repeated"] = castedLayer->get_ctc_merge_repeated() ? "1" : "0";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Erf>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Erf",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sign>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sign",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sin>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sin",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sinh>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sinh",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Asin>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Asin",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Cos>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Cos",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Cosh>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Cosh",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Acos>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Acos",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Tan>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Tan",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Atan>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Atan",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::Sqrt>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Sqrt",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::StridedSlice>::createLayer(
        const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "StridedSlice operation has a form that is not supported." << layer->get_friendly_name()
                       << " should be converted to StridedSliceIE operation";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::StridedSliceIE>::createLayer(
        const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "StridedSlice",
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::StridedSliceLayer>(params);
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::StridedSliceIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    std::string value;
    for (const auto& val : castedLayer->get_begin_mask()) {
        if (!value.empty()) value += ",";
        // plugins require reverse value of this mask.
        value += asString((1-val));
    }
    res->params["begin_mask"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_end_mask()) {
        if (!value.empty()) value += ",";
        // plugins require reverse value of this mask.
        value += asString((1-val));
    }
    res->params["end_mask"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_new_axis_mask()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["new_axis_mask"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_shrink_axis_mask()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["shrink_axis_mask"] = value;

    value.clear();
    for (const auto& val : castedLayer->get_ellipsis_mask()) {
        if (!value.empty()) value += ",";
        value += asString(val);
    }
    res->params["ellipsis_mask"] = value;

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
CNNLayer::Ptr NodeConverter<ngraph::op::HardSigmoid>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "HardSigmoid operation should be converted to HardSigmoid_IE";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::HardSigmoid_IE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = { layer->get_friendly_name(), "HardSigmoid", details::ngraph::convertPrecision(layer->get_output_element_type(0)) };
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
                          details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::GRN>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::GRNLayer>(params);
    res->params["bias"] = asString(castedLayer->get_bias());
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::LogicalNot>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "Activation", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::CNNLayer>(params);
    res->params["type"] = "not";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::ReduceLogicalAnd>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReduceAnd", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReduceLayer>(params);

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::v1::ReduceLogicalAnd>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["keep_dims"] = castedLayer->get_keep_dims() ? "True" : "False";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::ReduceLogicalOr>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "ReduceOr", details::ngraph::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::ReduceLayer>(params);

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::v1::ReduceLogicalOr>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    res->params["keep_dims"] = castedLayer->get_keep_dims() ? "True" : "False";
    return res;
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::v1::NonMaxSuppression>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    THROW_IE_EXCEPTION << "NonMaxSuppression operation must be converted to NonMaxSuppressionIE operation.";
}

template <>
CNNLayer::Ptr NodeConverter<ngraph::op::NonMaxSuppressionIE>::createLayer(const std::shared_ptr<ngraph::Node>& layer) const {
    LayerParams params = {layer->get_friendly_name(), "NonMaxSuppression", Precision::I32};

    auto castedLayer = std::dynamic_pointer_cast<ngraph::op::NonMaxSuppressionIE>(layer);
    if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << params.type << " layer " << params.name;

    auto res = std::make_shared<InferenceEngine::NonMaxSuppressionLayer>(params);
    res->params["sort_result_descending"] = std::to_string(castedLayer->m_sort_result_descending);
    res->params["center_point_box"] = std::to_string(castedLayer->m_sort_result_descending);
    return res;
}

}  // namespace Builder
}  // namespace InferenceEngine
