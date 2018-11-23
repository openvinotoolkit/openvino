// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "v2_layer_parsers.h"
#include "ie_cnn_net_reader_impl.h"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

CNNLayer::Ptr ActivationLayerCreator::CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms)  {
    pugi::xml_node dn = GetChild(node, { "data", "activation_data" }, false);
    if (dn.empty()) {
        THROW_IE_EXCEPTION << "Activation layer has no data node";
    }

    std::string type;
    for (auto ait = dn.attributes_begin(); ait != dn.attributes_end(); ++ait) {
        pugi::xml_attribute attr = *ait;
        if (CaselessEq<std::string>()("type", attr.name())) {
            if (!type.empty()) {
                THROW_IE_EXCEPTION << "Activation layer has multiple types";
            }
            type = attr.value();
        }
    }

    static caseless_map<std::string, std::shared_ptr<BaseCreator>> activationCreators = {
        {"relu", std::make_shared<V2LayerCreator<ReLULayer>>("ReLU")},
        {"prelu", std::make_shared<V2LayerCreator<PReLULayer>>("PReLU")},
        {"clamp", std::make_shared<V2LayerCreator<ClampLayer>>("Clamp")},
        {"elu", std::make_shared<V2LayerCreator<CNNLayer>>("ELU")},
        {"sigmoid", std::make_shared<V2LayerCreator<CNNLayer>>("Sigmoid")},
        {"tanh", std::make_shared<V2LayerCreator<CNNLayer>>("TanH")},
    };

    auto activationBuilder = activationCreators.find(type);
    if (activationBuilder == activationCreators.end()) {
        THROW_IE_EXCEPTION << "Unsupported Activation layer type: " << type;
    }

    auto activation = activationBuilder->second->CreateLayer(node, layerParsePrms);

    activation->type = activationBuilder->first;
    activation->params.erase("type");

    return activation;
}

CNNLayer::Ptr TILayerCreator::CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) {
    std::string ti_name = node.attribute("name").as_string();

    auto bn = node.child("body");
    if (bn.empty()) {
        THROW_IE_EXCEPTION << "TensorIterator " << ti_name << " has no body";
    }

    std::vector<TensorIterator::Port> _input_ports;
    std::vector<TensorIterator::Port> _output_ports;
    std::vector<TensorIterator::BackEdge> _backEdges;

    pugi::xml_node bedges = node.child("back_edges");
    FOREACH_CHILD(_ec, bedges, "edge") {
        int fromLayer = GetIntAttr(_ec, "from-layer");
        int fromPort = GetIntAttr(_ec, "from-port");
        int toLayer = GetIntAttr(_ec, "to-layer");
        int toPort = GetIntAttr(_ec, "to-port");

        _backEdges.push_back({ fromLayer, fromPort, toLayer, toPort });
    }

    pugi::xml_node ports = node.child("port_map");
    for (auto p = ports.first_child(); p; p = p.next_sibling()) {
        int external_port_id = GetIntAttr(p, "external_port_id");
        int internal_layer_id = GetIntAttr(p, "internal_layer_id");
        int internal_port_id = GetIntAttr(p, "internal_port_id");

        int axis = GetIntAttr(p, "axis", -1);
        int part_size = GetIntAttr(p, "part_size", -1);
        int stride = GetIntAttr(p, "stride", 0);

        TensorIterator::Port port{ external_port_id, internal_layer_id, internal_port_id, axis, part_size, stride };

        std::string pname(p.name());
        if ( pname == "input" ) {
            _input_ports.push_back(port);
        } else if (pname == "output") {
            _output_ports.push_back(port);
        } else {
            THROW_IE_EXCEPTION << "Unknown item {" << pname << "} in port map of TensorIterator " << ti_name;
        }
    }

    int prev_ir_version = BaseCreator::version_;
    auto pReader = std::make_shared<CNNNetReaderImpl>(std::make_shared<V2FormatParserCreator>());

    StatusCode status = pReader->ReadSubNetwork(bn);

    ResponseDesc resp;
    auto pNet = dynamic_cast<CNNNetworkImpl*>(pReader->getNetwork(&resp));

    bool recognized = false;
    unsigned axis_cand = 16;

    size_t layerCount = pNet->layerCount();

    if (layerCount == 3) {
        auto _layers = pNet->allLayers();

        for (auto &item : _layers) {
            auto cell = dynamic_cast<LSTMCell*> (item.second.get());

            if (cell != nullptr) {
                for (auto inputData : cell->insData) {
                    auto prevData = inputData.lock();
                    if (prevData == nullptr) {
                        THROW_IE_EXCEPTION << "No input reshape for LSTM cell " << cell->name;
                    }
                    auto inReshape  = dynamic_cast<ReshapeLayer*> (prevData->creatorLayer.lock().get());
                    auto outReshape = dynamic_cast<ReshapeLayer*> (cell->outData[0]->getInputTo().begin()->second.get());

                    if (inReshape != nullptr && outReshape != nullptr) {
                        layerParsePrms.prms.type = "RNN";
                        pReader->CopyBlobs(&layerParsePrms, cell->name);

                        // axis analysis
                        unsigned input_axis = _input_ports[0].axis;
                        size_t input_dims = layerParsePrms.inputPorts[0].dims.size();
                        unsigned output_axis = _output_ports[0].axis;
                        size_t output_dims = layerParsePrms.outputPorts[0].dims.size();
                        if ( input_axis == output_axis && input_dims == output_dims && input_axis < input_dims ) {
                            axis_cand = input_axis;
                        }

                        recognized = true;
                        break;
                    }
                }
                break;
            }
        }
    }

    // Global var. Need to restore after TI parsing.
    BaseCreator::version_ = prev_ir_version;

    if (recognized) {
        auto res = std::make_shared<RNNLayer>(layerParsePrms.prms);
        res->cellType = LSTM;

        /*** WA */
        {
            int d_ind = 0;
            int s1_ind = 0;
            int s2_ind = 0;
            if (_input_ports[1].internal_layer_id == _input_ports[2].internal_layer_id) {
                d_ind = 0; s1_ind = 1; s2_ind = 2;
            } else if (_input_ports[0].internal_layer_id == _input_ports[2].internal_layer_id) {
                d_ind = 1; s1_ind = 0; s2_ind = 2;
            } else if (_input_ports[0].internal_layer_id == _input_ports[1].internal_layer_id) {
                d_ind = 2; s1_ind = 0; s2_ind = 1;
            }
            res->params["swap_state"] = _input_ports[s1_ind].internal_port_id > _input_ports[s2_ind].internal_port_id ?
                    "YES" : "NO";
        }
        /*** end of WA */

        if (axis_cand < layerParsePrms.inputPorts[0].dims.size()) {
            res->_axis = axis_cand;
        }
        return res;
    } else {
        auto res = std::make_shared<TensorIterator>(layerParsePrms.prms);
        res->reader = pReader;
        res->input_ports = _input_ports;
        res->output_ports = _output_ports;
        res->backEdges = _backEdges;
        return res;
    }
}

