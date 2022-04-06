// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/precisions_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <iterator>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/layer_transformation.hpp"

using namespace ngraph;
using namespace ov;

PrecisionsAttribute::PrecisionsAttribute(const std::vector<ngraph::element::Type>& precisions) :
    SharedAttribute(precisions) {
}

ov::Any PrecisionsAttribute::create(
    const std::shared_ptr<ngraph::Node>& node,
    const AttributeParameters& params) {
    auto& rt = ov::is_type<opset1::FakeQuantize>(node) ? node->output(0).get_rt_info() : node->get_rt_info();
    return (rt[PrecisionsAttribute::get_type_info_static()] = PrecisionsAttribute(params.defaultPrecisions));
}

void PrecisionsAttribute::merge(std::vector<ov::Any>& attributes) {
    auto& my = value();
    for (auto attribute : attributes) {
        const auto& attributeValues = attribute.as<PrecisionsAttribute>().value();
        auto it = my.begin();
        while (it != my.end()) {
            if (std::find(attributeValues.begin(), attributeValues.end(), *it) == attributeValues.end()) {
                it = my.erase(it);
            } else {
                it++;
            }
        }
        if (my.size() == 0ul) {
            break;
        }
    }
}


std::string PrecisionsAttribute::to_string() const {
    std::stringstream ss;

    ss << attribute->get_string();

    bool firstPrecision = true;
    ss << "precisions: {";
    for (const auto& type : value()) {
        if (!firstPrecision) {
            ss << ", ";
        }
        ss << type;
        firstPrecision = false;
    }
    ss << "}";

    return ss.str();
}
