// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

using namespace InferenceEngine;

namespace vpu {

void FrontEnd::parseLogicalNot(const Model &model, const ie::CNNLayerPtr &layer, const DataVector &inputs, const DataVector &outputs) const {
    LayerParams params = {layer->name, "Eltwise", layer->precision};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->_operation = InferenceEngine::EltwiseLayer::Logical_NOT;

    parseEltwise(model, res, inputs, outputs);
}

void FrontEnd::parseAbs(const Model &model, const ie::CNNLayerPtr &layer, const DataVector &inputs, const DataVector &outputs) const {
    LayerParams params = {layer->name, "Eltwise", layer->precision};
    auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
    res->_operation = InferenceEngine::EltwiseLayer::Abs;

    parseEltwise(model, res, inputs, outputs);
}

void FrontEnd::parseActivation(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    const ie::details::caseless_map<std::string, LayerParser> activationParsers {
        {"not", LAYER_PARSER(parseLogicalNot)},
        {"abs", LAYER_PARSER(parseAbs)},
    };

    const auto type = layer->GetParamAsString("type");

    const auto activationParserIt = activationParsers.find(type);
    VPU_THROW_UNSUPPORTED_LAYER_UNLESS(activationParserIt != activationParsers.end(),
                                 "Failed to compile layer \"%v\"(type = %v) ", layer->name, type);

    activationParserIt->second(model, layer, inputs, outputs);
}

} // namespace vpu
