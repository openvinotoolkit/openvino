// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <ie_blob.h>
#include <ie_layers.h>

#include <builders/ie_layer_builder.hpp>
#include <details/caseless.hpp>
#include <ie_network.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {

namespace Builder {

IE_SUPPRESS_DEPRECATED_START

Layer builderFromCNNLayer(const CNNLayerPtr& cnnLayer);

struct ConvertersHolder {
    details::caseless_map<std::string, std::function<void(const CNNLayerPtr& cnnLayer, Layer&)>> converters;
};

/**
 * @brief This class registers layer validators
 */
class INFERENCE_ENGINE_API_CLASS(ConverterRegister) {
public:
    /**
     * @brief The constructor registers new layer validator
     * @param type Layer type
     * @param validator Layer validator
     */
    explicit ConverterRegister(const std::string& type,
                               const std::function<void(const CNNLayerPtr&, Layer&)>& converter);

    static void convert(const CNNLayerPtr& cnnLayer, Layer& layer) {
        if (getConvertersHolder().converters.find(layer.getType()) != getConvertersHolder().converters.end())
            getConvertersHolder().converters[layer.getType()](cnnLayer, layer);
    }

private:
    static ConvertersHolder& getConvertersHolder();
};

IE_SUPPRESS_DEPRECATED_END

#define REG_CONVERTER_FOR(__type, __converter) \
    static InferenceEngine::Builder::ConverterRegister _reg_converter_##__type(#__type, __converter)

template <class T>
inline std::string convertParameter2String(const Parameter& parameter) {
    if (parameter.is<std::vector<T>>()) {
        std::vector<T> params = parameter.as<std::vector<T>>();
        std::string result;
        for (const auto& param : params) {
            if (!result.empty()) result += ",";
            result += convertParameter2String<T>(param);
        }
        return result;
    }
    return std::to_string(parameter.as<T>());
}
template <>
inline std::string convertParameter2String<std::string>(const Parameter& parameter) {
    return parameter.as<std::string>();
}

INFERENCE_ENGINE_API_CPP(std::map<std::string, std::string>)
convertParameters2Strings(const std::map<std::string, Parameter>& parameters);

}  // namespace Builder
}  // namespace InferenceEngine
