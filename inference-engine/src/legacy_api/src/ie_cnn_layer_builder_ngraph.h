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

class INodeConverter {
public:
    virtual CNNLayer::Ptr createLayer(const std::shared_ptr<ngraph::Node>& layer) const = 0;
    virtual bool canCreate(const std::shared_ptr<ngraph::Node>& node) const = 0;
    virtual ~INodeConverter() = default;
};

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
std::string asString<double>(const double& value);

template <>
std::string asString<float>(const float& value);

template <class NGT>
class NodeConverter : public INodeConverter {
public:
    NodeConverter() = default;
    ~NodeConverter() override = default;

    CNNLayer::Ptr createLayer(const std::shared_ptr<ngraph::Node>& layer) const override;

    bool canCreate(const std::shared_ptr<ngraph::Node>& node) const override {
        auto castedPtr = ngraph::as_type_ptr<NGT>(node);
        return castedPtr != nullptr;
    }
};

}  // namespace Builder
}  // namespace InferenceEngine
