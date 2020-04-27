// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <ie_blob.h>
#include <ie_layers.h>
#include <ie_parameter.hpp>

#include <details/caseless.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {

namespace Builder {

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

}  // namespace Builder
}  // namespace InferenceEngine
