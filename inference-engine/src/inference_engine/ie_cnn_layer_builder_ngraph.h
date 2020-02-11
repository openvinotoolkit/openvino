// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <ie_blob.h>
#include <ie_layers.h>

#include <details/caseless.hpp>
#include <ie_network.hpp>
#include <map>
#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/op/constant.hpp>
#include <string>
#include <vector>

#include "blob_factory.hpp"
#include "ie_ngraph_utils.hpp"

namespace InferenceEngine {

namespace Builder {

class INodeConverter {
public:
    virtual CNNLayer::Ptr createLayer(const std::shared_ptr<ngraph::Node>& layer) const = 0;
    virtual bool canCreate(const std::shared_ptr<ngraph::Node>& node) const = 0;

    template <class T>
    static std::string asString(const T& value) {
        return std::to_string(value);
    }

    template <class T>
    static std::string asString(const std::vector<T>& value);
};

template <typename T>
std::string INodeConverter::asString(const std::vector<T>& value) {
    std::string result;
    for (const auto& item : value) {
        if (!result.empty()) result += ",";
        result += asString(item);
    }
    return result;
}

template <class NGT>
class NodeConverter : public INodeConverter {
public:
    NodeConverter() = default;

    CNNLayer::Ptr createLayer(const std::shared_ptr<ngraph::Node>& layer) const override;

    bool canCreate(const std::shared_ptr<ngraph::Node>& node) const override {
        auto castedPtr = std::dynamic_pointer_cast<NGT>(node);
        return castedPtr != nullptr;
    }

private:
    Blob::Ptr shareWeights(const std::shared_ptr<ngraph::op::Constant>& constLayer) const {
        if (!constLayer) THROW_IE_EXCEPTION << "Cannot share weights! Constant operation is empty!";
        auto dataPrecision = details::ngraph::convertPrecision(constLayer->get_element_type());

        size_t shapeSize = ngraph::shape_size(constLayer->get_shape());
        if (dataPrecision == Precision::BIN) {
            shapeSize = (shapeSize % 8 == 0 ? shapeSize / 8 : (shapeSize / 8) + 1);
        }

        TensorDesc td(dataPrecision, {shapeSize}, Layout::C);
        return make_blob_with_precision(td, const_cast<void *>(constLayer->get_data_ptr()));
    }
};

}  // namespace Builder
}  // namespace InferenceEngine
