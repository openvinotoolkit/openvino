// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "v2_layer_parsers.h"

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



