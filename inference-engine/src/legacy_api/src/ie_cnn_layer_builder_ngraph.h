// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <ngraph/node.hpp>
#include <ngraph/op/constant.hpp>

#include <ie_api.h>
#include <ie_blob.h>
#include "blob_factory.hpp"

#include <legacy/ie_layers.h>
#include <ie_ngraph_utils.hpp>

namespace InferenceEngine {

namespace Builder {

template <class T>
std::string asString(const T& value) {
    return std::to_string(value);
}

template <typename T>
std::string asString(const std::vector<T>& value) {
    std::string result;
    for (const auto& item : value) {
        if (!result.empty()) result += ",";
        result += asString(item);
    }
    return result;
}

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

}  // namespace Builder
}  // namespace InferenceEngine
