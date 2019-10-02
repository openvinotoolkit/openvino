// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <details/caseless.hpp>
#include <ie_network.hpp>
#include <ie_layers.h>
#include <ie_blob.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include <ngraph/node.hpp>

namespace InferenceEngine {

namespace Builder {

class INodeConverter {
public:
    virtual CNNLayer::Ptr createLayer(const std::shared_ptr<ngraph::Node>& layer, const Precision &precision) const = 0;
    virtual bool canCreate(const std::shared_ptr<ngraph::Node>& node) const = 0;

    template <class T>
    static std::string asString(const T& value) {
        return std::to_string(value);
    }
};

template <class NGT>
class NodeConverter: public INodeConverter {
public:
    NodeConverter() = default;

    CNNLayer::Ptr createLayer(const std::shared_ptr<ngraph::Node>& layer, const Precision &precision) const override;

    bool canCreate(const std::shared_ptr<ngraph::Node>& node) const override {
        auto castedPtr = std::dynamic_pointer_cast<NGT>(node);
        return castedPtr != nullptr;
    }
};

}  // namespace Builder
}  // namespace InferenceEngine
